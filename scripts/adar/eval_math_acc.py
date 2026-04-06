"""
评估模型的数学解题正确率.
读取verl main_generation输出的parquet, 提取boxed答案并与ground truth对比.

用法:
    # 从生成的parquet评估
    python eval_math_acc.py --gen_parquet <生成结果parquet> --test_data <原始测试数据parquet> --label <标签>
"""

import argparse
import json
import re
import sys
import os

import pandas as pd


def extract_boxed(text):
    """从模型输出中提取答案, 按优先级尝试多种格式."""
    # 1. \boxed{}
    matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
    if matches:
        return matches[-1].strip()
    # 2. <answer>...</answer>
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 3. #### 数字
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if match:
        return match.group(1).replace(',', '').strip()
    return None


def extract_last_num(text):
    """提取字符串中的最后一个数字."""
    text = str(text).replace(",", "")
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches:
        return float(matches[-1])
    return None


def check_answer(model_answer, expected_answer):
    """检查答案是否正确."""
    if model_answer is None:
        return False
    model_num = extract_last_num(model_answer)
    expected_num = extract_last_num(expected_answer)
    if model_num is not None and expected_num is not None:
        return abs(model_num - expected_num) < 1e-2
    return model_answer.strip() == str(expected_answer).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_parquet", required=True, help="main_generation输出的parquet")
    parser.add_argument("--test_data", default=None, help="原始测试数据parquet (用于获取ground truth)")
    parser.add_argument("--label", default="", help="评估标签 (用于输出)")
    parser.add_argument("--output", default=None, help="输出结果JSON路径")
    args = parser.parse_args()

    # 加载生成结果
    gen_df = pd.read_parquet(args.gen_parquet)
    print(f"---EVAL--- [{args.label}] 加载生成结果: {args.gen_parquet}, {len(gen_df)} 条")
    print(f"---EVAL--- 列: {list(gen_df.columns)}")

    # 获取ground truth
    if args.test_data:
        test_df = pd.read_parquet(args.test_data)
        answers = []
        for _, row in test_df.iterrows():
            extra = row["extra_info"]
            answers.append(extra["answer"])
    elif "reward_model" in gen_df.columns:
        answers = []
        for _, row in gen_df.iterrows():
            rm = row["reward_model"]
            if isinstance(rm, dict):
                answers.append(rm.get("ground_truth", ""))
            else:
                answers.append(str(rm))
    elif "extra_info" in gen_df.columns:
        answers = [row["extra_info"]["answer"] for _, row in gen_df.iterrows()]
    else:
        print("---ERROR--- 无法获取ground truth, 请指定--test_data")
        sys.exit(1)

    # 找到responses列
    response_key = None
    for key in ["responses", "response", "generated_text", "output"]:
        if key in gen_df.columns:
            response_key = key
            break
    if response_key is None:
        print(f"---ERROR--- 找不到response列, 可用列: {list(gen_df.columns)}")
        sys.exit(1)

    print(f"---EVAL--- response列: {response_key}")

    # 评估
    correct = 0
    no_boxed = 0
    no_think_close = 0
    total = min(len(gen_df), len(answers))
    details = []

    for i in range(total):
        response = gen_df.iloc[i][response_key]
        if isinstance(response, list):
            response = response[0]  # 取第一个sample
        response = str(response)
        expected = answers[i]

        # 统计
        if "</think>" not in response:
            no_think_close += 1

        # 剥离<think>
        clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
        extracted = extract_boxed(clean)
        if extracted is None:
            no_boxed += 1

        is_correct = check_answer(extracted, expected)
        if is_correct:
            correct += 1

        details.append({
            "idx": i,
            "expected": expected,
            "extracted": extracted,
            "correct": is_correct,
            "response_len": len(response),
            "think_closed": "</think>" in response,
        })

    # 汇总
    print(f"\n{'='*60}")
    print(f"---EVAL--- [{args.label}] 结果:")
    print(f"---EVAL--- 总题数: {total}")
    print(f"---EVAL--- 正确: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"---EVAL--- 无boxed答案: {no_boxed}/{total} ({no_boxed/total*100:.1f}%)")
    print(f"---EVAL--- think未闭合: {no_think_close}/{total} ({no_think_close/total*100:.1f}%)")
    print(f"{'='*60}")

    # 打印部分错误case
    wrong = [d for d in details if not d["correct"]]
    print(f"\n---EVAL--- 部分错误case (前10):")
    for d in wrong[:10]:
        print(f"  题{d['idx']}: expected={d['expected']}, extracted={d['extracted']}, "
              f"think_closed={d['think_closed']}, resp_len={d['response_len']}")

    if args.output:
        summary = {
            "label": args.label,
            "gen_parquet": args.gen_parquet,
            "total": total,
            "correct": correct,
            "accuracy": correct / total,
            "no_boxed": no_boxed,
            "no_think_close": no_think_close,
            "details": details,
        }
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"---EVAL--- 结果保存到: {args.output}")


if __name__ == "__main__":
    main()
