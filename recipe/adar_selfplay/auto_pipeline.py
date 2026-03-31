"""
自动校验+扰动流水线 (auto_pipeline.py)

将AdaR原有的scripts中的核心逻辑重构为可import的函数:
- parse_and_verify: 从模型生成结果中解析模板和代码, 并校验正确性
- perturb_variables: 对通过校验的样本进行变量扰动
- check_evs: 检查模型解答是否与代码运行结果一致 (EVS验证)

这些函数在trainer的driver进程上以CPU执行, 不需要GPU.
"""

import re
import os
import sys
import time
import random
import logging
import contextlib
import multiprocessing
import itertools
from io import StringIO
from queue import Empty
from fractions import Fraction
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# SafeExecutor: 在子进程中安全执行Python代码
# ============================================================

class SafeExecutor:
    """
    安全代码执行器: 在常驻子进程中执行Python代码, 支持超时和重启.
    用于校验模型生成的代码是否可执行以及结果是否正确.
    """

    def __init__(self):
        self._task_q = multiprocessing.Queue()
        self._result_q = multiprocessing.Queue()
        self._req_id_iter = itertools.count()
        self._start_worker()

    def _start_worker(self):
        """启动一个常驻子进程, 负责执行代码"""
        self._proc = multiprocessing.Process(
            target=self._worker,
            args=(self._task_q, self._result_q),
        )
        self._proc.daemon = True
        self._proc.start()

    @staticmethod
    def _worker(task_q, result_q):
        """子进程: 设置builtins + 自定义print + exec代码, 返回结果字符串"""
        import builtins as _builtins
        _builtins.input = lambda prompt="": ""
        _builtins.quit = lambda *a, **kw: ""
        _builtins.exit = lambda *a, **kw: ""

        while True:
            item = task_q.get()
            if item is None:
                break
            req_id, code = item
            local_env = {'__result__': ""}

            def custom_print(*args, **kwargs):
                sep = kwargs.get('sep', ' ')
                end = kwargs.get('end', '\n')
                output = sep.join(str(arg) for arg in args) + end
                local_env['__result__'] += output

            local_env['print'] = custom_print
            try:
                exec(code, local_env)
                result_q.put((req_id, "ok", local_env["__result__"]))
            except Exception as e:
                result_q.put((req_id, "error", repr(e)))

    def _restart_worker(self):
        """在超时或崩溃时重启子进程"""
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._start_worker()

    def run(self, code: str, timeout: float = 2.0) -> Optional[str]:
        """
        执行代码并返回print输出的字符串.
        超时或出错返回None.
        """
        req_id = next(self._req_id_iter)
        self._task_q.put((req_id, code))

        start = time.time()
        while True:
            remaining = timeout - (time.time() - start)
            if remaining <= 0:
                self._restart_worker()
                return None
            try:
                rid, status, payload = self._result_q.get(timeout=remaining)
            except Empty:
                self._restart_worker()
                return None
            if rid != req_id:
                continue
            if status == "error":
                return None
            return payload.strip()

    def run_as_number(self, code: str, timeout: float = 2.0) -> Optional[float]:
        """执行代码并尝试将输出转为float. 非数字或出错返回None."""
        result = self.run(code, timeout=timeout)
        if result is None:
            return None
        try:
            return float(result)
        except ValueError:
            if re.search(r'\d', result):
                return None
            return None

    def close(self):
        """优雅关闭子进程"""
        try:
            self._task_q.put(None)
        except Exception:
            pass
        if self._proc.is_alive():
            self._proc.join(timeout=5)


# ============================================================
# 通用工具函数
# ============================================================

def extract_last_num(text) -> float:
    """从文本中提取最后一个数字. 用于答案比对."""
    if isinstance(text, (int, float)):
        return float(text)
    text = str(text)
    text = re.sub(r"(\d),(\d)", r"\g<1>\g<2>", text)
    res = re.findall(r"\\boxed\{(\d+(\.\d+)?)", text)
    if len(res) == 0:
        res = re.findall(r"(\d+(\.\d+)?)", text)
    if len(res) > 0:
        return float(res[-1][0])
    else:
        return 0.0


def extract_last_number_from_solution(text: str) -> Optional[float]:
    """从模型解答中提取最后一个数字, 优先从\\boxed{}中提取."""
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    # 先尝试boxed
    idx = text.rfind("\\boxed")
    if idx >= 0:
        i = idx
        num_left = 0
        right_idx = None
        while i < len(text):
            if text[i] == "{":
                num_left += 1
            if text[i] == "}":
                num_left -= 1
                if num_left == 0:
                    right_idx = i
                    break
            i += 1
        if right_idx is not None:
            start = text.index("{", idx) + 1
            boxed_content = text[start:right_idx]
            nums = re.findall(r"(-?\d+\.?\d*)", boxed_content)
            if nums:
                try:
                    return float(nums[-1])
                except ValueError:
                    pass
    # fallback: 找最后一个数字
    nums = re.findall(r"(-?\d+\.?\d*)", text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    return None


def check_validity(value, old_value):
    """检查扰动后的执行结果是否有效"""
    try:
        if isinstance(old_value, str):
            return old_value.strip().lower() == str(value).strip().lower()
        old_value = int(old_value) if float(old_value).is_integer() else float(old_value)
        value = float(value)
        if isinstance(old_value, int):
            if value.is_integer():
                value = int(value)
            else:
                return False
        return value * old_value >= 0
    except (ValueError, TypeError):
        return False


# ============================================================
# Step 1: 解析模板和代码 (parse_and_verify)
# ============================================================

def parse_and_verify(
    generation: str,
    query: str,
    answer: str,
    executor: SafeExecutor,
    code_timeout: float = 2.0,
) -> Optional[dict]:
    """
    从模型的一次生成结果中解析模板和Python代码, 并验证正确性.

    Args:
        generation: 模型生成的文本 (应包含### Template:和### Python Code:)
        query: 原始数学题
        answer: 原始答案
        executor: SafeExecutor实例
        code_timeout: 代码执行超时(秒)

    Returns:
        成功返回 {"template": str, "python": str, "answer": answer}
        失败返回 None
    """
    # 提取模板
    template_match = re.search(
        r'### (?:Query|Query Template|Template):(.*?)(?=###|$)',
        generation, re.DOTALL | re.IGNORECASE
    )
    template_content = template_match.group(1).strip() if template_match else None

    # 提取Python代码
    python_code_match = re.search(
        r'### Python Code:\s*```(?:python)?\s*(.*?)\s*```',
        generation, re.DOTALL | re.IGNORECASE
    )
    python_code = python_code_match.group(1).strip() if python_code_match else None

    if python_code is None:
        logger.debug("---PARSE--- FAIL: 无法提取Python代码")
        return None
    if template_content is None:
        logger.debug("---PARSE--- FAIL: 无法提取模板")
        return None

    # 检查模板与原始问题的相似度 (空格数差异不超过50%)
    if abs(template_content.count(' ') - query.count(' ')) > 0.5 * min(query.count(' ') + 1, template_content.count(' ') + 1):
        logger.debug("---PARSE--- FAIL: 模板与原始问题不匹配")
        return None

    # 执行代码
    python_result = executor.run_as_number(python_code, timeout=code_timeout)
    if python_result is None:
        logger.debug("---PARSE--- FAIL: 代码运行错误/超时")
        return None

    # 检查执行结果与答案是否一致
    expected = extract_last_num(answer)
    if abs(python_result - expected) > 1e-2:
        logger.debug(f"---PARSE--- FAIL: 代码结果={python_result}, 答案={expected}")
        return None

    # 检查模板变量与代码变量是否对齐
    variables = re.findall(r'<([^>]+?)>', template_content)
    for var in variables:
        pattern = r'\b' + re.escape(var) + r'\s*?='
        if re.search(pattern, python_code) is None:
            logger.debug(f"---PARSE--- FAIL: 变量'{var}'在代码中未找到定义")
            return None

    logger.debug(f"---PARSE--- SUCCESS: 模板变量={variables}")
    return {
        "template": template_content,
        "python": python_code,
        "answer": answer,
    }


# ============================================================
# Step 2: 变量扰动 (perturb_variables)
# ============================================================

def randomize_value(original_value, max_fluct: float = 1.0, upper_bound: float = 10**9, retry_times: int = 10):
    """
    对原始值进行随机扰动, 范围为 ±(max_fluct * original_value).
    保持类型(int/float)不变.
    """
    lower = original_value * (1 - max_fluct)
    if original_value > 0:
        lower = max(1 if isinstance(original_value, int) else 0.01, lower)
    upper = min(original_value * (1 + max_fluct), upper_bound)
    if original_value < 0:
        upper = min(-1 if isinstance(original_value, int) else -0.01, upper)
    if isinstance(original_value, float) and 0 < original_value < 1:
        lower = max(0.01, lower)
        upper = min(0.99, upper)

    if random.random() < 0.5 and original_value != 0:
        for _ in range(retry_times):
            new_value = random.uniform(lower, upper)
            if isinstance(original_value, int):
                new_value = int(round(new_value))
            else:
                new_value = round(new_value, 2)
            if new_value != original_value:
                break
            max_fluct += 0.1
            lower = original_value * (1 - max_fluct)
            if original_value > 0:
                lower = max(1 if isinstance(original_value, int) else 0.01, lower)
            upper = original_value * (1 + max_fluct)
            if original_value < 0:
                upper = min(-1 if isinstance(original_value, int) else -0.01, upper)
            if isinstance(original_value, float) and 0 < original_value < 1:
                lower = max(0.01, lower)
                upper = min(0.99, upper)
    else:
        new_value = random.uniform(lower, upper)
        if isinstance(original_value, int):
            new_value = int(round(new_value))
        else:
            new_value = round(new_value, 2)

    return new_value


def randomize_code_once(
    original_code: str,
    original_query: str,
    original_ans: float,
    max_fluct: float,
    executor: SafeExecutor,
    code_timeout: float = 2.0,
    retry_times: int = 10,
) -> Optional[dict]:
    """
    对一段代码做一次变量扰动, 并验证结果有效性.

    Returns:
        成功返回 {"new_query": str, "new_code": str, "new_ans": float, "max_fluct": float}
        失败返回 None
    """
    lines = original_code.split('\n')

    # 收集变量定义行 (代码开头到第一个空行之间)
    variable_lines = []
    break_idx = 0
    for i, line in enumerate(lines):
        if not variable_lines and line.strip() == "":
            continue
        if line.strip() == "":
            break_idx = i
            break
        variable_lines.append(line)
        break_idx = i + 1

    pattern = re.compile(r'^(\s*\w+)\s*=\s*(\d[\d/ ]*|\d*\.\d*)\s*(#.*)?$')
    pattern2 = re.compile(r'^(\s*\w+)\s*=\s*(.*?)\s*(#.*)?$')

    # 检查模板对齐: 无占位符的变量视为常量, 跳过不扰动
    # (修复: 代码中可能包含不可扰动的常量如 days_in_week=7, 不应拒绝整个样本)
    skipped_consts = set()
    for line in variable_lines:
        match = pattern.match(line)
        if match:
            prefix = match.group(1).strip()
            if f"<{prefix}>" not in original_query:
                skipped_consts.add(prefix)
    if skipped_consts:
        print(f"---PERTURB--- 跳过无占位符的常量: {skipped_consts}")

    # 统计可扰动变量数 (排除无占位符的常量)
    variable_num = sum(
        1 for line in variable_lines
        if pattern.match(line) and pattern.match(line).group(1).strip() not in skipped_consts
    )
    if variable_num == 0:
        print(f"---PERTURB--- FAIL: 无可扰动变量 (全部为常量: {skipped_consts})")
        return None

    for variable_limits in range(variable_num, 0, -1):
        for _ in range(retry_times):
            variable_count = 0
            new_variable_lines = []
            replaced_variables = []

            for line in variable_lines:
                match = pattern.match(line)
                if match:
                    prefix = match.group(1)
                    original_value_str = match.group(2)
                    suffix = match.group(3) or ""

                    # 无占位符的常量: 保持原值不扰动
                    if prefix.strip() in skipped_consts:
                        new_value_str = original_value_str
                        replaced_variables.append((prefix.strip(), str(new_value_str)))
                        new_variable_lines.append(prefix + " = " + new_value_str + " " + suffix)
                        continue

                    if variable_limits == variable_count:
                        new_value_str = original_value_str
                    else:
                        if '/' in original_value_str:
                            original_value_str_clean = original_value_str.replace('//', '/').replace(' ', '')
                            try:
                                original_value = Fraction(original_value_str_clean)
                            except Exception:
                                new_variable_lines.append(line)
                                continue
                            numerator = original_value.numerator
                            denominator = original_value.denominator
                            if numerator == 1:
                                denominator = randomize_value(denominator, max_fluct=max_fluct)
                            else:
                                numerator = randomize_value(numerator, max_fluct=max_fluct)
                            new_value_str = f"{numerator}/{denominator}"
                        else:
                            if abs(float(original_value_str) - int(float(original_value_str))) < 1e-6:
                                original_value = int(float(original_value_str))
                            else:
                                original_value = float(original_value_str)
                            new_val = randomize_value(
                                original_value,
                                max_fluct=max_fluct,
                                upper_bound=100 if 'percentage' in prefix else 10**9,
                            )
                            new_value_str = str(new_val)

                    replaced_variables.append((prefix.strip(), str(new_value_str)))
                    new_variable_lines.append(prefix + " = " + new_value_str + " " + suffix)
                    variable_count += 1
                else:
                    new_variable_lines.append(line)
                    match2 = pattern2.match(line)
                    if match2:
                        replaced_variables.append((match2.group(1).strip(), str(match2.group(2))))

            final_code = '\n'.join(new_variable_lines) + '\n' + '\n'.join(lines[break_idx:])

            result_str = executor.run(final_code, timeout=code_timeout)
            if result_str is None:
                continue
            try:
                result_val = float(result_str)
            except (ValueError, TypeError):
                continue

            if check_validity(result_val, original_ans):
                # 替换query模板中的变量
                new_query = original_query
                for var, new_value in replaced_variables:
                    new_query = new_query.replace(f"<{var}>", str(new_value))

                return {
                    "new_query": new_query,
                    "new_code": final_code,
                    "new_ans": result_val,
                    "max_fluct": max_fluct,
                }

    return None


def perturb_variables(
    template: str,
    python_code: str,
    answer,
    executor: SafeExecutor,
    n_perturbations: int = 5,
    alpha_list: list = None,
    timeout_total: float = 180.0,
    code_timeout: float = 2.0,
    retry_times: int = 10,
) -> list[dict]:
    """
    对一个已校验的样本进行多次变量扰动.

    Args:
        template: 问题模板 (含<var>占位符)
        python_code: 解题代码
        answer: 原始答案
        executor: SafeExecutor实例
        n_perturbations: 目标扰动次数
        alpha_list: 扰动幅度列表, 默认[5]
        timeout_total: 总超时时间
        code_timeout: 单次代码执行超时
        retry_times: 内部重试次数

    Returns:
        扰动结果列表, 每个元素为 {"new_query", "new_code", "new_ans", "max_fluct"}
    """
    if alpha_list is None:
        alpha_list = [5]

    original_ans = extract_last_num(answer)
    results = []
    start_time = time.time()

    for max_fluct in alpha_list:
        per_alpha_count = n_perturbations // len(alpha_list)
        for _ in range(per_alpha_count):
            if time.time() - start_time > timeout_total:
                logger.info(f"---PERTURB--- 总超时, 已完成 {len(results)} 次扰动")
                break

            r = randomize_code_once(
                original_code=python_code,
                original_query=template,
                original_ans=original_ans,
                max_fluct=max_fluct,
                executor=executor,
                code_timeout=code_timeout,
                retry_times=retry_times,
            )
            if r is not None:
                results.append(r)

    return results


# ============================================================
# Step 3: EVS验证 (check_evs)
# ============================================================

def check_evs(
    model_responses: list[str],
    expected_answer: float,
    tolerance: float = 1e-3,
) -> tuple[bool, Optional[str]]:
    """
    检查模型的多个解答中是否有至少一个与代码运行结果一致.

    Args:
        model_responses: 模型的多个解答文本
        expected_answer: 代码运行得到的正确答案
        tolerance: 数值比较容差

    Returns:
        (is_passed, best_response): 是否通过EVS, 以及通过的那个response
    """
    for response in model_responses:
        extracted = extract_last_number_from_solution(response)
        if extracted is not None and abs(extracted - expected_answer) < tolerance:
            return True, response
    return False, None


def compute_evs_accuracy(
    model_responses: list[str],
    expected_answer: float,
    tolerance: float = 1e-3,
) -> float:
    """
    计算模型多个解答中正确解答的比例 (accuracy).
    用于T3的reward计算.
    """
    if not model_responses:
        return 0.0
    correct = 0
    for response in model_responses:
        extracted = extract_last_number_from_solution(response)
        if extracted is not None and abs(extracted - expected_answer) < tolerance:
            correct += 1
    return correct / len(model_responses)
