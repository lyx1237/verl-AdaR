"""
RayAdaRSelfPlayTrainer (adar_selfplay_ray_trainer.py)

自定义Trainer, 继承RayPPOTrainer, 实现AdaR Self-Play的4阶段训练流水线.

4个阶段:
  Stage1: 生成模板+代码 → 自动校验+扰动
  Stage2: 解答扰动问题 → EVS筛选
  Stage3: Paraphrase → 构造新的问题
  Stage4: 解答paraphrase后的问题 → 计算reward

阶段启用控制 (通过配置):
  - enable_selfplay=False时, 只运行T4 (等同于标准GRPO)
  - enable_selfplay=True时, 运行全部Stage1~Stage4
  - enable_stage3_paraphrase 控制是否启用Stage3+Stage4 (关闭则只做Stage1+Stage2)

所有阶段共享同一个actor模型, 4个阶段的loss按权重加权后一起更新.
"""

import os
import uuid
import logging
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.debug import marked_timer

from .prompt_builder import PromptBuilder
from .auto_pipeline import SafeExecutor, parse_and_verify, perturb_variables, check_evs, compute_evs_accuracy
from .adar_selfplay_reward import (
    compute_stage1_reward,
    compute_stage2_reward,
    compute_stage3_reward,
    compute_stage4_reward,
)

logger = logging.getLogger(__name__)


class RayAdaRSelfPlayTrainer(RayPPOTrainer):
    """
    AdaR Self-Play Trainer.

    在标准GRPO训练循环基础上, 支持4阶段的self-play pipeline.
    """

    def _init_selfplay_components(self):
        """初始化self-play相关的组件"""
        # PromptBuilder: 用于为各阶段构造不同格式的prompt
        self.prompt_builder = PromptBuilder(
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
        )

        # SafeExecutor: 用于安全执行模型生成的Python代码
        self.executor = SafeExecutor()

        # 读取self-play配置
        sp_cfg = self.config.get("adar_selfplay", {})
        self.enable_selfplay = sp_cfg.get("enable_selfplay", False)
        self.enable_stage3_paraphrase = sp_cfg.get("enable_stage3_paraphrase", True)

        # rollout次数
        self.n1 = sp_cfg.get("n1", 4)   # Stage1: 模板+代码生成
        self.n2 = sp_cfg.get("n2", 5)   # 扰动次数 (非模型)
        self.n3 = sp_cfg.get("n3", 8)   # Stage2: 解答扰动题
        self.n4 = sp_cfg.get("n4", 4)   # Stage3: paraphrase
        self.n5 = sp_cfg.get("n5", 8)   # Stage4: 解答paraphrase题

        # loss权重
        self.w1 = sp_cfg.get("w1", 0.2)  # Stage1 loss权重
        self.w2 = sp_cfg.get("w2", 0.3)  # Stage2 loss权重
        self.w3 = sp_cfg.get("w3", 0.2)  # Stage3 loss权重
        self.w4 = sp_cfg.get("w4", 0.3)  # Stage4 loss权重

        # 扰动参数
        self.perturb_alpha = sp_cfg.get("perturb_alpha", 5)
        self.perturb_timeout = sp_cfg.get("perturb_timeout", 30)
        self.code_timeout = sp_cfg.get("code_timeout", 2.0)

        # 各阶段最大序列长度
        self.max_template_code_length = sp_cfg.get("max_template_code_length", 2048)
        self.max_solve_length = sp_cfg.get("max_solve_length", 2048)
        self.max_paraphrase_length = sp_cfg.get("max_paraphrase_length", 1024)

        # Debug: 注入假的T1通过结果, 用于测试完整pipeline
        self.debug_inject_stage1 = sp_cfg.get("debug_inject_stage1", False)

        mode_str = "Self-Play (Stage1~Stage4)" if self.enable_selfplay else "Stage4-Only (标准GRPO)"
        logger.info(f"---SELFPLAY--- 模式: {mode_str}")
        if self.debug_inject_stage1:
            logger.info("---SELFPLAY--- ⚠ DEBUG模式: 注入假T1结果, 仅供测试!")
            print("---SELFPLAY--- ⚠ DEBUG模式: 注入假T1结果, 仅供测试!")
        if self.enable_selfplay:
            logger.info(f"---SELFPLAY--- T3_Paraphrase: {self.enable_stage3_paraphrase}")
            logger.info(f"---SELFPLAY--- rollout次数: n1={self.n1}, n2={self.n2}, n3={self.n3}, n4={self.n4}, n5={self.n5}")
            logger.info(f"---SELFPLAY--- loss权重: w1={self.w1}, w2={self.w2}, w3={self.w3}, w4={self.w4}")
        print(f"---SELFPLAY--- 模式: {mode_str}")

    def _decode_responses(self, batch: DataProto) -> list[str]:
        """将batch中的responses token ids解码为文本"""
        responses = batch.batch["responses"]  # (batch_size, response_len)
        texts = []
        for i in range(responses.shape[0]):
            resp_ids = responses[i].tolist()
            # 去掉padding token
            resp_ids = [t for t in resp_ids if t != self.tokenizer.pad_token_id]
            text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            texts.append(text)
        return texts

    def _debug_make_fake_t1(self, queries, answers, n_prompts):
        """
        DEBUG: 为每个问题生成假的T1通过结果.
        - rollout 0: 所有问题都通过 (有效的template+code)
        - rollout 1: 只有偶数问题通过 (测试per-rollout逻辑)
        """
        # 通用模板: 用简单的乘法/加法模板, 变量可被perturb_variables扰动
        fake_templates = {
            # query含 "X dollars each" + "Y apples" → price*count
            "apples": ("A store sells apples for <price> dollars each. If Tom buys <count> apples, how much does he pay in total?",
                       "price = {p}\ncount = {c}\nresult = price * count\nprint(result)"),
            # length*width
            "rectangle": ("A rectangle has a length of <length> cm and a width of <width> cm. What is the area of the rectangle?",
                          "length = {p}\nwidth = {c}\nresult = length * width\nprint(result)"),
            # a - b
            "candies": ("Sarah has <total> candies. She gives <give> candies to her friend. How many candies does Sarah have left?",
                        "total = {p}\ngive = {c}\nresult = total - give\nprint(result)"),
            # speed * time
            "car": ("A car travels at a speed of <speed> km/h for <hours> hours. How far does it travel?",
                    "speed = {p}\nhours = {c}\nresult = speed * hours\nprint(result)"),
            # boxes * per_box
            "boxes": ("John has <boxes> boxes. Each box contains <per_box> pencils. How many pencils does John have in total?",
                      "boxes = {p}\nper_box = {c}\nresult = boxes * per_box\nprint(result)"),
            # 0.5 * base * height
            "triangle": ("A triangle has a base of <base> meters and a height of <height> meters. What is its area?",
                         "base = {p}\nheight = {c}\nresult = 0.5 * base * height\nprint(result)"),
            # per_day * days
            "baker": ("A baker makes <per_day> cakes per day. How many cakes does the baker make in <days> days?",
                      "per_day = {p}\ndays = {c}\nresult = per_day * days\nprint(result)"),
            # price * (1 - discount)
            "shirt": ("If a shirt costs <price> dollars and there is a <discount_pct> percent discount, what is the final price?",
                      "price = {p}\ndiscount_pct = {c}\nresult = price * (1 - discount_pct / 100)\nprint(result)"),
        }

        # 关键词→模板的映射
        keyword_map = [
            ("apples", "apples", 3, 5),
            ("rectangle", "rectangle", 8, 6),
            ("candies", "candies", 20, 7),
            ("speed", "car", 60, 3),
            ("boxes", "boxes", 4, 12),
            ("triangle", "triangle", 10, 6),
            ("baker", "baker", 15, 4),
            ("shirt", "shirt", 25, 20),
        ]

        stage1_passed = {}
        for p_idx in range(n_prompts):
            q = queries[p_idx].lower()
            # 找匹配的模板
            matched = None
            for keyword, tpl_key, param_p, param_c in keyword_map:
                if keyword in q:
                    template, code_tpl = fake_templates[tpl_key]
                    code = code_tpl.format(p=param_p, c=param_c)
                    matched = {"template": template, "python": code}
                    break

            if matched is None:
                # 默认: 用一个通用模板
                ans_val = float(answers[p_idx]) if answers[p_idx] else 0
                matched = {
                    "template": queries[p_idx],
                    "python": f"result = {ans_val}\nprint(result)",
                }

            # rollout 0: 所有问题都通过
            stage1_passed[(p_idx, 0)] = matched

            # rollout 1: 只有偶数问题通过 (测试per-rollout差异)
            if p_idx % 2 == 0:
                stage1_passed[(p_idx, 1)] = matched

        return stage1_passed

    def _generate_for_stage(
        self,
        prompt_batch: DataProto,
        n_rollouts: int,
        stage_name: str,
        timing_raw: dict,
    ) -> DataProto:
        """
        为某个阶段调用generate_sequences, 并返回生成结果.

        Args:
            prompt_batch: 该阶段的输入DataProto (未repeat)
            n_rollouts: 每个prompt的rollout次数
            stage_name: 阶段名称, 用于日志
            timing_raw: 计时字典

        Returns:
            包含生成结果的DataProto (已repeat+union)
        """
        logger.info(f"---{stage_name}--- 开始生成, batch_size={len(prompt_batch)}, n={n_rollouts}")
        print(f"---{stage_name}--- 开始生成, batch_size={len(prompt_batch)}, n={n_rollouts}")

        # repeat prompt以匹配rollout次数
        gen_input = prompt_batch.repeat(repeat_times=n_rollouts, interleave=True)

        with marked_timer(f"gen_{stage_name}", timing_raw, color="red"):
            gen_output = self.actor_rollout_wg.generate_sequences(gen_input)
            if "timing" in gen_output.meta_info:
                timing_raw.update(gen_output.meta_info["timing"])
                gen_output.meta_info.pop("timing", None)

        # gen_output包含完整的input_ids(prompt+response), attention_mask, position_ids等
        # 将gen_input的non_tensor_batch合并到gen_output (不包含tensor, 避免input_ids冲突)
        result = gen_output
        for key in gen_input.non_tensor_batch:
            if key not in result.non_tensor_batch:
                result.non_tensor_batch[key] = gen_input.non_tensor_batch[key]

        logger.info(f"---{stage_name}--- 生成完成, 总responses={len(result)}")
        print(f"---{stage_name}--- 生成完成, 总responses={len(result)}")

        return result

    def _run_selfplay_pipeline(
        self,
        batch_dict: dict,
        metrics: dict,
        timing_raw: dict,
    ) -> list[DataProto]:
        """
        运行完整的Self-Play pipeline, 返回各阶段的batch列表.

        每个阶段的batch已经包含了reward (token_level_scores)和advantage.

        Returns:
            stage_batches: [stage1_batch, stage2_batch, stage3_batch, stage4_batch]
            其中未启用的阶段对应None
        """
        # 从原始数据中获取queries和answers
        original_batch = DataProto.from_single_dict(batch_dict)
        # 从data_source和reward_model等获取信息
        queries = []
        answers = []
        responses_for_t1 = []  # chosen responses, 用于T1的输入

        # 获取原始数据
        for i in range(len(original_batch)):
            item = original_batch[i]
            # 从non_tensor_batch获取reward_model信息
            if "reward_model" in item.non_tensor_batch:
                rm_info = item.non_tensor_batch["reward_model"]
                if isinstance(rm_info, dict):
                    answers.append(str(rm_info.get("ground_truth", "0")))
                else:
                    answers.append(str(rm_info))
            else:
                answers.append("0")

            # 从extra_info获取原始query
            if "extra_info" in item.non_tensor_batch:
                extra = item.non_tensor_batch["extra_info"]
                if isinstance(extra, dict):
                    queries.append(extra.get("query", ""))
                    responses_for_t1.append(extra.get("chosen", ""))
                else:
                    queries.append("")
                    responses_for_t1.append("")
            else:
                queries.append("")
                responses_for_t1.append("")

        n_prompts = len(queries)
        logger.info(f"---SELFPLAY--- 开始pipeline, {n_prompts}个原始问题")
        print(f"---SELFPLAY--- 开始pipeline, {n_prompts}个原始问题")

        stage_batches = [None, None, None, None]  # Stage1, Stage2, Stage3, Stage4

        # ====== Stage 1: 生成模板+代码 (Stage1) ======
        # 每个rollout独立评估, 不再只取第一个通过的
        stage1_batch = None
        # (p_idx, r_idx) -> {"template": ..., "python": ...} 通过parse_and_verify的rollout
        stage1_passed_rollouts = {}

        with marked_timer("stage1", timing_raw, color="blue"):
            logger.info("---STAGE1--- Stage1: 生成模板和代码")
            print("---STAGE1--- Stage1: 生成模板和代码")

            stage1_prompt_batch = self.prompt_builder.build_stage1_prompts(
                queries=queries,
                responses=responses_for_t1,
                max_length=self.max_template_code_length,
            )

            stage1_batch = self._generate_for_stage(
                stage1_prompt_batch, self.n1, "Stage1", timing_raw
            )

            stage1_responses = self._decode_responses(stage1_batch)

            # 对每个rollout独立检查parse_and_verify
            if self.debug_inject_stage1:
                # DEBUG模式: 注入假T1结果
                # rollout 0 对所有问题都通过, rollout 1 只对偶数问题通过
                stage1_passed_rollouts = self._debug_make_fake_t1(queries, answers, n_prompts)
                print(f"---DEBUG--- 注入假T1结果: {len(stage1_passed_rollouts)} rollouts通过")
            else:
                for p_idx in range(n_prompts):
                    for r_idx in range(self.n1):
                        flat_idx = p_idx * self.n1 + r_idx
                        if flat_idx >= len(stage1_responses):
                            break
                        result = parse_and_verify(
                            generation=stage1_responses[flat_idx],
                            query=queries[p_idx],
                            answer=answers[p_idx],
                            executor=self.executor,
                            code_timeout=self.code_timeout,
                        )
                        if result is not None:
                            stage1_passed_rollouts[(p_idx, r_idx)] = {
                                "template": result["template"],
                                "python": result["python"],
                            }

            k1 = len(stage1_passed_rollouts)
            n_prompts_with_pass = len(set(k[0] for k in stage1_passed_rollouts))
            logger.info(f"---STAGE1--- Stage1校验通过: {k1}/{n_prompts * self.n1} rollouts "
                        f"(来自 {n_prompts_with_pass}/{n_prompts} 个问题)")
            print(f"---STAGE1--- Stage1校验通过: {k1}/{n_prompts * self.n1} rollouts")
            metrics["selfplay/stage1_verify_pass_rate"] = k1 / max(n_prompts * self.n1, 1)
            metrics["selfplay/stage1_verify_prompt_rate"] = n_prompts_with_pass / max(n_prompts, 1)

        # ====== Stage 1.5: 自动扰动 (对每个通过的Stage1 rollout分别扰动) ======
        # (p_idx, r_idx) -> list of perturbation dicts
        perturbed_data = {}

        with marked_timer("stage1.5_perturb", timing_raw, color="blue"):
            logger.info("---STAGE1.5--- 自动扰动")
            print("---STAGE1.5--- 自动扰动")

            for (p_idx, r_idx), rollout_result in stage1_passed_rollouts.items():
                perturbations = perturb_variables(
                    template=rollout_result["template"],
                    python_code=rollout_result["python"],
                    answer=answers[p_idx],
                    executor=self.executor,
                    n_perturbations=self.n2,
                    alpha_list=[self.perturb_alpha],
                    timeout_total=self.perturb_timeout,
                    code_timeout=self.code_timeout,
                )
                if perturbations:
                    perturbed_data[(p_idx, r_idx)] = perturbations

            logger.info(f"---STAGE1.5--- 扰动完成: {len(perturbed_data)}/{k1} rollouts有扰动")
            print(f"---STAGE1.5--- 扰动完成: {len(perturbed_data)}/{k1} rollouts有扰动")
            metrics["selfplay/perturb_pass_rate"] = len(perturbed_data) / max(k1, 1)

        # ====== Stage 2: 解答扰动问题 + EVS筛选 (Stage2) ======
        stage2_batch = None
        stage2_expected_answers = []
        # (p_idx, r_idx) -> list of perturbation dicts that passed EVS
        passed_perturbations = {}

        if perturbed_data:
            with marked_timer("stage2", timing_raw, color="green"):
                logger.info("---STAGE2--- Stage2: 解答扰动问题 + EVS筛选")
                print("---STAGE2--- Stage2: 解答扰动问题 + EVS筛选")

                stage2_queries = []
                stage2_codes = []
                stage2_expected_answers = []
                stage2_source_keys = []  # (p_idx, r_idx) for each perturbation

                for (p_idx, r_idx), perturbations in perturbed_data.items():
                    for pert in perturbations:
                        stage2_queries.append(pert["new_query"])
                        stage2_codes.append(pert["new_code"])
                        stage2_expected_answers.append(pert["new_ans"])
                        stage2_source_keys.append((p_idx, r_idx))

                if stage2_queries:
                    stage2_prompt_batch = self.prompt_builder.build_stage2_prompts(
                        queries=stage2_queries,
                        codes=stage2_codes,
                        max_length=self.max_solve_length,
                    )

                    stage2_batch = self._generate_for_stage(
                        stage2_prompt_batch, self.n3, "Stage2", timing_raw
                    )

                    # EVS筛选: 对每个扰动问题, 检查n3个回答中是否有正确的
                    stage2_responses = self._decode_responses(stage2_batch)

                    for q_idx in range(len(stage2_queries)):
                        responses_for_q = []
                        for r_idx_inner in range(self.n3):
                            flat_idx = q_idx * self.n3 + r_idx_inner
                            if flat_idx < len(stage2_responses):
                                responses_for_q.append(stage2_responses[flat_idx])

                        passed, _ = check_evs(
                            model_responses=responses_for_q,
                            expected_answer=stage2_expected_answers[q_idx],
                        )

                        if passed:
                            key = stage2_source_keys[q_idx]
                            if key not in passed_perturbations:
                                passed_perturbations[key] = []
                            passed_perturbations[key].append({
                                "query": stage2_queries[q_idx],
                                "code": stage2_codes[q_idx],
                                "answer": stage2_expected_answers[q_idx],
                            })

                    total_passed_perts = sum(len(v) for v in passed_perturbations.values())
                    logger.info(f"---STAGE2--- EVS通过: {total_passed_perts}/{len(stage2_queries)} 扰动, "
                                f"涉及 {len(passed_perturbations)}/{len(perturbed_data)} rollouts")
                    print(f"---STAGE2--- EVS通过: {total_passed_perts}/{len(stage2_queries)} 扰动")
                    metrics["selfplay/evs_pass_rate"] = total_passed_perts / max(len(stage2_queries), 1)
                    metrics["selfplay/evs_rollout_rate"] = len(passed_perturbations) / max(len(perturbed_data), 1)

        # ====== Stage 3: Paraphrase (Stage3) ======
        # 收集ALL通过EVS的扰动问题, 不再只取每个prompt的第一个
        stage3_batch = None
        paraphrased_questions = []  # list of (question_text, expected_answer)
        stage3_expected_answers = []  # 用于T4的expected answers

        if self.enable_stage3_paraphrase and passed_perturbations:
            with marked_timer("stage3", timing_raw, color="yellow"):
                logger.info("---STAGE3--- Stage3: Paraphrase")
                print("---STAGE3--- Stage3: Paraphrase")

                # 收集ALL通过EVS的扰动问题 (a个)
                stage3_questions = []
                stage3_expected_answers = []

                for key, perts in passed_perturbations.items():
                    for pert in perts:
                        stage3_questions.append(pert["query"])
                        stage3_expected_answers.append(pert["answer"])

                a_total = len(stage3_questions)
                logger.info(f"---STAGE3--- 共{a_total}个通过EVS的扰动进入paraphrase")
                print(f"---STAGE3--- 共{a_total}个通过EVS的扰动进入paraphrase")

                if stage3_questions:
                    stage3_prompt_batch = self.prompt_builder.build_stage3_prompts(
                        questions=stage3_questions,
                        max_length=self.max_paraphrase_length,
                    )

                    stage3_batch = self._generate_for_stage(
                        stage3_prompt_batch, self.n4, "Stage3", timing_raw
                    )

                    # 解码paraphrase结果, 收集ALL非空paraphrase进入T4
                    stage3_responses = self._decode_responses(stage3_batch)

                    # stage3_valid_map: t4问题index -> stage3_batch中对应的(q_idx, r_idx)
                    stage3_valid_map = []
                    for q_idx in range(len(stage3_questions)):
                        for r_idx in range(self.n4):
                            flat_idx = q_idx * self.n4 + r_idx
                            if flat_idx < len(stage3_responses) and stage3_responses[flat_idx].strip():
                                paraphrased_questions.append((
                                    stage3_responses[flat_idx].strip(),
                                    stage3_expected_answers[q_idx],
                                ))
                                stage3_valid_map.append((q_idx, r_idx))

                    b_total = len(paraphrased_questions)
                    logger.info(f"---STAGE3--- Paraphrase完成: {b_total}个有效变体问题 "
                                f"(来自{a_total}个扰动 x {self.n4} rollouts)")
                    print(f"---STAGE3--- Paraphrase完成: {b_total}个有效变体问题")
                    metrics["selfplay/paraphrase_count"] = b_total
        else:
            stage3_valid_map = []

        # ====== Stage 4: 解答paraphrase后的问题 (Stage4) ======
        stage4_batch = None

        if paraphrased_questions:
            with marked_timer("stage4", timing_raw, color="red"):
                logger.info(f"---STAGE4--- Stage4: 解答{len(paraphrased_questions)}个问题")
                print(f"---STAGE4--- Stage4: 解答{len(paraphrased_questions)}个问题")

                stage4_questions = [q for q, _ in paraphrased_questions]

                stage4_prompt_batch = self.prompt_builder.build_stage4_prompts(
                    questions=stage4_questions,
                    max_length=self.max_solve_length,
                )

                stage4_batch = self._generate_for_stage(
                    stage4_prompt_batch, self.n5, "Stage4", timing_raw
                )

                logger.info(f"---STAGE4--- Stage4生成完成")
                print(f"---STAGE4--- Stage4生成完成")

        # ====== 计算各阶段reward ======
        logger.info("---REWARD--- 开始计算各阶段reward")
        print("---REWARD--- 开始计算各阶段reward")

        # Stage1 reward (per-rollout: 基于EVS结果)
        if stage1_batch is not None:
            stage1_rewards = compute_stage1_reward(
                n_prompts=n_prompts,
                n1=self.n1,
                stage1_passed_rollouts=stage1_passed_rollouts,
                passed_perturbations=passed_perturbations,
            )
            seq_len = stage1_batch.batch["attention_mask"].shape[1]
            stage1_batch.batch["token_level_scores"] = torch.zeros(len(stage1_batch), seq_len)
            stage1_batch.batch["token_level_scores"][:, -1] = stage1_rewards
            stage1_batch.batch["token_level_rewards"] = stage1_batch.batch["token_level_scores"].clone()
            metrics["selfplay/stage1_avg_reward"] = stage1_rewards.mean().item()

        # Stage4 reward (先计算, 因为Stage3 reward依赖Stage4的accuracy)
        # Stage4: 全错的group masked out
        stage4_accuracies = []
        if stage4_batch is not None:
            stage4_responses = self._decode_responses(stage4_batch)
            stage4_expected = [a for _, a in paraphrased_questions]
            stage4_reward_scores, stage4_accuracies, stage4_training_mask = compute_stage4_reward(
                responses=stage4_responses,
                expected_answers=stage4_expected,
                group_size=self.n5,
            )
            seq_len = stage4_batch.batch["attention_mask"].shape[1]
            stage4_batch.batch["token_level_scores"] = torch.zeros(len(stage4_batch), seq_len)
            for i in range(len(stage4_batch)):
                stage4_batch.batch["token_level_scores"][i, -1] = stage4_reward_scores[i]
            stage4_batch.batch["token_level_rewards"] = stage4_batch.batch["token_level_scores"].clone()
            # 存储training_mask, 在advantage计算时应用
            # 存储1D per-sample mask, 在advantage计算时再expand到seq_len
            stage4_batch.batch["training_mask"] = stage4_training_mask
            metrics["selfplay/stage4_avg_reward"] = stage4_reward_scores.mean().item()
            metrics["selfplay/stage4_avg_accuracy"] = np.mean(stage4_accuracies) if stage4_accuracies else 0.0
            metrics["selfplay/stage4_masked_out_groups"] = sum(1 for acc in stage4_accuracies if acc == 0.0)

        # Stage2 reward: 全错的group masked out
        if stage2_batch is not None:
            stage2_responses = self._decode_responses(stage2_batch)
            stage2_reward_scores, group_has_correct, stage2_training_mask = compute_stage2_reward(
                responses=stage2_responses,
                expected_answers=stage2_expected_answers,
                group_size=self.n3,
            )
            seq_len = stage2_batch.batch["attention_mask"].shape[1]
            stage2_batch.batch["token_level_scores"] = torch.zeros(len(stage2_batch), seq_len)
            for i in range(len(stage2_batch)):
                stage2_batch.batch["token_level_scores"][i, -1] = stage2_reward_scores[i]
            stage2_batch.batch["token_level_rewards"] = stage2_batch.batch["token_level_scores"].clone()
            # 存储training_mask
            stage2_batch.batch["training_mask"] = stage2_training_mask
            metrics["selfplay/stage2_avg_reward"] = stage2_reward_scores.mean().item()
            metrics["selfplay/stage2_masked_out_groups"] = sum(1 for x in group_has_correct if not x)

        # Stage3 reward (依赖Stage4 accuracy, 通过stage3_valid_map映射)
        if stage3_batch is not None and stage4_accuracies:
            # 将Stage4 accuracy映射回T3的每个paraphrase rollout
            # stage3_valid_map[j] = (q_idx, r_idx): 第j个Stage4问题对应t3的第q_idx个源问题的第r_idx个rollout
            # stage4_accuracies[j]: 第j个Stage4问题的准确率
            # Stage3 reward: 对stage3_batch中每个entry, 找到其对应的Stage4 accuracy
            n_stage3_sources = len(stage3_expected_answers)  # a: 通过EVS的扰动数
            stage3_reward_scores = torch.zeros(len(stage3_batch))

            # 为每个t3 entry找到对应的Stage4 accuracy
            # stage3_batch有 n_stage3_sources * n4 个entry
            # stage3_valid_map告诉我们哪些t3 entry产生了有效的Stage4问题
            stage4_acc_by_stage3_entry = {}  # (q_idx, r_idx) -> stage4_accuracy
            for j, (q_idx, r_idx) in enumerate(stage3_valid_map):
                if j < len(stage4_accuracies):
                    stage4_acc_by_stage3_entry[(q_idx, r_idx)] = stage4_accuracies[j]

            for q_idx in range(n_stage3_sources):
                for r_idx in range(self.n4):
                    flat_idx = q_idx * self.n4 + r_idx
                    if flat_idx >= len(stage3_batch):
                        break
                    if (q_idx, r_idx) in stage4_acc_by_stage3_entry:
                        acc = stage4_acc_by_stage3_entry[(q_idx, r_idx)]
                        reward = max(0.0, 1.0 - 4.0 * (acc - 0.5) ** 2)
                        stage3_reward_scores[flat_idx] = reward
                    # else: 空paraphrase, reward=0

            seq_len = stage3_batch.batch["attention_mask"].shape[1]
            stage3_batch.batch["token_level_scores"] = torch.zeros(len(stage3_batch), seq_len)
            for i in range(len(stage3_batch)):
                stage3_batch.batch["token_level_scores"][i, -1] = stage3_reward_scores[i]
            stage3_batch.batch["token_level_rewards"] = stage3_batch.batch["token_level_scores"].clone()
            metrics["selfplay/stage3_avg_reward"] = stage3_reward_scores.mean().item()

        # ====== 详细日志: 用于手动验证reward正确性 ======
        self._log_detailed_reward_debug(
            n_prompts=n_prompts,
            queries=queries,
            answers=answers,
            stage1_passed_rollouts=stage1_passed_rollouts,
            stage1_rewards=stage1_rewards if stage1_batch is not None else None,
            perturbed_data=perturbed_data,
            passed_perturbations=passed_perturbations,
            stage2_expected_answers=stage2_expected_answers,
            stage2_responses=self._decode_responses(stage2_batch) if stage2_batch is not None else [],
            stage2_reward_scores=stage2_reward_scores if stage2_batch is not None else None,
            stage2_training_mask=stage2_training_mask if stage2_batch is not None else None,
            group_has_correct=group_has_correct if stage2_batch is not None else [],
            stage3_valid_map=stage3_valid_map if stage3_batch is not None else [],
            paraphrased_questions=paraphrased_questions,
            stage4_responses=self._decode_responses(stage4_batch) if stage4_batch is not None else [],
            stage4_accuracies=stage4_accuracies,
            stage4_reward_scores=stage4_reward_scores if stage4_batch is not None else None,
            stage4_training_mask=stage4_training_mask if stage4_batch is not None else None,
            stage3_reward_scores=stage3_reward_scores if (stage3_batch is not None and stage4_accuracies) else None,
        )

        stage_batches = [stage1_batch, stage2_batch, stage3_batch, stage4_batch]
        return stage_batches

    def _log_detailed_reward_debug(self, **kwargs):
        """输出详细的per-sample日志, 用于手动验证reward."""
        sep = "─" * 80
        print(f"\n{'═' * 80}")
        print(f"  DETAILED REWARD DEBUG DUMP (手动验证用)")
        print(f"{'═' * 80}")

        n_prompts = kwargs["n_prompts"]
        queries = kwargs["queries"]
        answers = kwargs["answers"]
        stage1_passed = kwargs["stage1_passed_rollouts"]
        stage1_rewards = kwargs["stage1_rewards"]
        perturbed_data = kwargs["perturbed_data"]
        passed_perts = kwargs["passed_perturbations"]

        # === Stage1 ===
        print(f"\n{sep}")
        print(f"  Stage1 REWARD (per-rollout)")
        print(f"{sep}")
        if stage1_rewards is not None:
            for p_idx in range(n_prompts):
                q_short = queries[p_idx][:60] + "..." if len(queries[p_idx]) > 60 else queries[p_idx]
                print(f"  问题{p_idx}: {q_short}  (ans={answers[p_idx]})")
                for r_idx in range(self.n1):
                    flat_idx = p_idx * self.n1 + r_idx
                    key = (p_idx, r_idx)
                    verify = "✓" if key in stage1_passed else "✗"
                    has_pert = key in perturbed_data
                    has_evs = key in passed_perts and len(passed_perts[key]) > 0
                    reward = stage1_rewards[flat_idx].item() if flat_idx < len(stage1_rewards) else 0
                    n_evs_pass = len(passed_perts[key]) if key in passed_perts else 0
                    print(f"    rollout{r_idx}: verify={verify}, has_pert={has_pert}, "
                          f"evs_pass={n_evs_pass}, reward={reward:.0f}")

        # === Stage2 ===
        stage2_expected = kwargs["stage2_expected_answers"]
        stage2_responses = kwargs["stage2_responses"]
        stage2_scores = kwargs["stage2_reward_scores"]
        stage2_mask = kwargs["stage2_training_mask"]
        ghc = kwargs["group_has_correct"]

        if stage2_scores is not None and len(stage2_expected) > 0:
            print(f"\n{sep}")
            print(f"  Stage2 REWARD + MASK (per-perturbation, n3={self.n3})")
            print(f"{sep}")
            from .auto_pipeline import extract_last_number_from_solution
            for q_idx in range(len(stage2_expected)):
                has_correct = ghc[q_idx] if q_idx < len(ghc) else False
                mask_val = stage2_mask[q_idx * self.n3].item() if stage2_mask is not None else 1
                print(f"  扰动{q_idx}: expected={stage2_expected[q_idx]}, "
                      f"has_correct={'✓' if has_correct else '✗'}, mask={mask_val:.0f}")
                for r_idx in range(self.n3):
                    flat_idx = q_idx * self.n3 + r_idx
                    if flat_idx < len(stage2_responses):
                        resp_short = stage2_responses[flat_idx][:80].replace('\n', ' ')
                        extracted = extract_last_number_from_solution(stage2_responses[flat_idx])
                        reward = stage2_scores[flat_idx].item()
                        print(f"    尝试{r_idx}: extracted={extracted}, reward={reward:.0f}, "
                              f"resp=\"{resp_short}\"")

        # === Stage3 + Stage4 ===
        stage3_valid_map = kwargs["stage3_valid_map"]
        paraphrased = kwargs["paraphrased_questions"]
        stage4_responses = kwargs["stage4_responses"]
        stage4_acc = kwargs["stage4_accuracies"]
        stage4_scores = kwargs["stage4_reward_scores"]
        stage4_mask = kwargs["stage4_training_mask"]
        stage3_scores = kwargs["stage3_reward_scores"]

        if stage4_scores is not None and len(paraphrased) > 0:
            print(f"\n{sep}")
            print(f"  Stage4 REWARD + MASK (per-paraphrase, n5={self.n5})")
            print(f"{sep}")
            from .auto_pipeline import extract_last_number_from_solution
            for j in range(len(paraphrased)):
                q_text, expected_ans = paraphrased[j]
                acc = stage4_acc[j] if j < len(stage4_acc) else 0
                mask_val = stage4_mask[j * self.n5].item() if stage4_mask is not None else 1
                stage3_entry = stage3_valid_map[j] if j < len(stage3_valid_map) else "?"
                q_short = q_text[:60].replace('\n', ' ')
                print(f"  Stage4问题{j} (from stage3 entry {stage3_entry}): "
                      f"expected={expected_ans}, acc={acc:.2f}, mask={mask_val:.0f}")
                print(f"    paraphrase: \"{q_short}\"")
                for r_idx in range(self.n5):
                    flat_idx = j * self.n5 + r_idx
                    if flat_idx < len(stage4_responses):
                        extracted = extract_last_number_from_solution(stage4_responses[flat_idx])
                        reward = stage4_scores[flat_idx].item()
                        resp_short = stage4_responses[flat_idx][:80].replace('\n', ' ')
                        print(f"    回答{r_idx}: extracted={extracted}, reward={reward:.0f}, "
                              f"resp=\"{resp_short}\"")

        if stage3_scores is not None:
            print(f"\n{sep}")
            print(f"  Stage3 REWARD (per-paraphrase-rollout)")
            print(f"{sep}")
            for i in range(len(stage3_scores)):
                r = stage3_scores[i].item()
                mapped = "空" if r == 0.0 and (i // self.n4, i % self.n4) not in dict(
                    [(k, v) for v, k in enumerate(stage3_valid_map)] if stage3_valid_map else []
                ) else f"{r:.4f}"
                print(f"    stage3_batch[{i}]: reward={mapped}")

        print(f"{'═' * 80}\n")

    def _compute_advantage_for_stage(
        self,
        batch: DataProto,
        n_rollouts: int,
        stage_name: str,
        timing_raw: dict,
    ) -> DataProto:
        """为单个阶段计算response_mask, old_log_probs和advantage"""
        if batch is None:
            return None

        logger.info(f"---ADV--- 计算{stage_name}的advantage, batch_size={len(batch)}")

        # response_mask
        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        # old_log_probs (使用当前actor计算)
        with marked_timer(f"old_log_prob_{stage_name}", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            if "entropys" in old_log_prob.batch:
                old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # reference policy log_prob (如果使用KL loss)
        if self.use_reference_policy:
            with marked_timer(f"ref_{stage_name}", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute advantage (GRPO)
        with marked_timer(f"adv_{stage_name}", timing_raw, color="brown"):
            norm_adv_by_std = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=n_rollouts,
                norm_adv_by_std_in_grpo=norm_adv_by_std,
            )

        # 应用training_mask: 将被mask的样本的response_mask置0, 使其不参与参数更新
        if "training_mask" in batch.batch:
            training_mask_1d = batch.batch["training_mask"]  # (batch_size,)
            # expand到与response_mask相同的seq_len维度
            seq_len = batch.batch["response_mask"].shape[1]
            training_mask = training_mask_1d.unsqueeze(1).expand(-1, seq_len)
            batch.batch["response_mask"] = batch.batch["response_mask"] * training_mask
            # 同时将masked样本的advantage置0
            if "advantages" in batch.batch:
                batch.batch["advantages"] = batch.batch["advantages"] * training_mask
            masked_count = int((training_mask_1d == 0).sum().item())
            logger.info(f"---ADV--- {stage_name}: training_mask排除了{masked_count}/{len(batch)}个样本")
            print(f"---ADV--- {stage_name}: training_mask排除了{masked_count}/{len(batch)}个样本")
            # 清理training_mask, 后续不再需要
            batch.batch.pop("training_mask", None)

        return batch

    def _merge_and_update(
        self,
        stage_batches: list,
        stage_names: list[str],
        stage_weights: list[float],
        stage_n_rollouts: list[int],
        metrics: dict,
        timing_raw: dict,
    ):
        """
        合并多个阶段的batch, 调整advantage权重, 然后一次性update_actor.

        通过缩放advantage来实现加权loss:
          advantage_i *= weight_i
        """
        # 过滤掉None的阶段
        valid_batches = []
        valid_weights = []
        valid_names = []

        for batch, name, weight in zip(stage_batches, stage_names, stage_weights):
            if batch is not None and len(batch) > 0:
                valid_batches.append(batch)
                valid_weights.append(weight)
                valid_names.append(name)

        if not valid_batches:
            logger.warning("---UPDATE--- 没有有效的batch可以更新!")
            print("---UPDATE--- 没有有效的batch可以更新!")
            return

        logger.info(f"---UPDATE--- 合并阶段: {valid_names}, 权重: {valid_weights}")
        print(f"---UPDATE--- 合并阶段: {valid_names}, 权重: {valid_weights}")

        # 缩放advantage
        for batch, weight, name in zip(valid_batches, valid_weights, valid_names):
            if "advantages" in batch.batch:
                batch.batch["advantages"] = batch.batch["advantages"] * weight
                logger.info(f"---UPDATE--- {name} advantage缩放 x{weight}, "
                           f"mean={batch.batch['advantages'].mean().item():.4f}")

        # 合并所有batch (需要先pad到相同序列长度)
        if len(valid_batches) == 1:
            combined_batch = valid_batches[0]
        else:
            # 不同阶段可能有不同的序列长度, 需要按key分别pad到最大长度
            # 收集每个key在所有batch中的最大dim-1
            from tensordict import TensorDict as TD
            all_keys = list(valid_batches[0].batch.keys())
            max_dim1 = {}
            for key in all_keys:
                for b in valid_batches:
                    t = b.batch[key]
                    if t.dim() >= 2:
                        max_dim1[key] = max(max_dim1.get(key, 0), t.shape[1])

            for idx, b in enumerate(valid_batches):
                new_data = {}
                needs_pad = False
                for key in all_keys:
                    tensor = b.batch[key]
                    if key in max_dim1 and tensor.dim() >= 2 and tensor.shape[1] < max_dim1[key]:
                        pad_len = max_dim1[key] - tensor.shape[1]
                        padding = torch.zeros(
                            (tensor.shape[0], pad_len, *tensor.shape[2:]),
                            dtype=tensor.dtype, device=tensor.device,
                        )
                        new_data[key] = torch.cat([tensor, padding], dim=1)
                        needs_pad = True
                    else:
                        new_data[key] = tensor
                if needs_pad:
                    b.batch = TD(new_data, batch_size=[new_data[all_keys[0]].shape[0]])
                    print(f"---UPDATE--- {valid_names[idx]} padded to match max lengths")

            # 清除non_tensor_batch (不同阶段可能有不兼容的结构, update_actor不需要)
            for b in valid_batches:
                b.non_tensor_batch = {}
            combined_batch = DataProto.concat(valid_batches)

        logger.info(f"---UPDATE--- 合并后batch_size={len(combined_batch)}")
        print(f"---UPDATE--- 合并后batch_size={len(combined_batch)}")

        # balance batch (如果配置了)
        if self.config.trainer.balance_batch:
            self._balance_batch(combined_batch, metrics=metrics)

        combined_batch.meta_info["global_token_num"] = torch.sum(
            combined_batch.batch["attention_mask"], dim=-1
        ).tolist()

        # update actor
        with marked_timer("update_actor", timing_raw, color="red"):
            combined_batch.meta_info["multi_turn"] = False
            actor_output = self.actor_rollout_wg.update_actor(combined_batch)
        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
        metrics.update(actor_output_metrics)

        logger.info(f"---UPDATE--- Actor更新完成")
        print(f"---UPDATE--- Actor更新完成")

    def fit(self):
        """
        主训练循环.

        支持两种模式:
        1. enable_selfplay=False: 标准GRPO训练 (只用T4)
        2. enable_selfplay=True: Self-Play 4阶段pipeline
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        # 初始化self-play组件
        self._init_selfplay_components()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # 加载checkpoint
        self._load_checkpoint()

        # 训练前验证
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = defaultdict(float)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    if self.enable_selfplay:
                        # ====== Self-Play模式: 4阶段pipeline ======
                        stage_batches = self._run_selfplay_pipeline(
                            batch_dict, metrics, timing_raw
                        )

                        # 为每个阶段计算advantage
                        stage1_batch, stage2_batch, stage3_batch, stage4_batch = stage_batches
                        n_rollouts_list = [self.n1, self.n3, self.n4, self.n5]
                        names = ["Stage1", "Stage2", "Stage3", "Stage4"]

                        for idx, (batch, n_roll, name) in enumerate(
                            zip(stage_batches, n_rollouts_list, names)
                        ):
                            if batch is not None:
                                stage_batches[idx] = self._compute_advantage_for_stage(
                                    batch, n_roll, name, timing_raw
                                )

                        # 合并并更新
                        self._merge_and_update(
                            stage_batches=stage_batches,
                            stage_names=names,
                            stage_weights=[self.w1, self.w2, self.w3, self.w4],
                            stage_n_rollouts=n_rollouts_list,
                            metrics=metrics,
                            timing_raw=timing_raw,
                        )
                    else:
                        # ====== 标准GRPO模式 (Stage4-only) ======
                        self._run_standard_grpo(batch_dict, metrics, timing_raw)

                # 验证
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 保存checkpoint
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # 收集metrics
                timing_metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
                metrics.update(timing_metrics)

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    # 清理executor
                    if hasattr(self, 'executor'):
                        self.executor.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1

        # 清理
        if hasattr(self, 'executor'):
            self.executor.close()

    def _run_standard_grpo(self, batch_dict: dict, metrics: dict, timing_raw: dict):
        """
        标准GRPO训练流程 (等同于原版verl PPO训练, 但使用AdaR的reward函数).
        当enable_selfplay=False时使用此路径.
        """
        batch = DataProto.from_single_dict(batch_dict)

        # 添加uid
        batch.non_tensor_batch["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )

        gen_batch = self._get_gen_batch(batch)
        gen_batch.meta_info["global_steps"] = self.global_steps
        n = self.config.actor_rollout_ref.rollout.n

        gen_batch_output = gen_batch.repeat(repeat_times=n, interleave=True)

        # 生成
        with marked_timer("gen", timing_raw, color="red"):
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
            if "timing" in gen_batch_output.meta_info:
                timing_raw.update(gen_batch_output.meta_info["timing"])
                gen_batch_output.meta_info.pop("timing", None)

        batch = batch.repeat(repeat_times=n, interleave=True)
        batch = batch.union(gen_batch_output)

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)

        # balance batch
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

        # 计算reward
        with marked_timer("reward", timing_raw, color="yellow"):
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        # old_log_probs
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            entropys = old_log_prob.batch.get("entropys")
            if entropys is not None:
                response_masks = batch.batch["response_mask"]
                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                metrics["actor/entropy"] = entropy_agg.detach().item()
                old_log_prob.batch.pop("entropys")
            batch = batch.union(old_log_prob)

        # reference policy
        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)

        # compute advantage
        with marked_timer("adv", timing_raw, color="brown"):
            norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
                num_repeat=n,
                norm_adv_by_std_in_grpo=norm_adv,
            )

        # update critic
        if self.use_critic:
            with marked_timer("update_critic", timing_raw, color="pink"):
                critic_output = self.critic_wg.update_critic(batch)
            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
            metrics.update(critic_output_metrics)

        # update actor
        if self.config.trainer.critic_warmup <= self.global_steps:
            with marked_timer("update_actor", timing_raw, color="red"):
                batch.meta_info["multi_turn"] = False
                actor_output = self.actor_rollout_wg.update_actor(batch)
            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
            metrics.update(actor_output_metrics)

        # data metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
