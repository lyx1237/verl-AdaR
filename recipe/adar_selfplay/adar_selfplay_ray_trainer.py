"""
RayAdaRSelfPlayTrainer (adar_selfplay_ray_trainer.py)

自定义Trainer, 继承RayPPOTrainer, 实现AdaR Self-Play的4阶段训练流水线.

4个阶段:
  T1: 生成模板+代码 → 自动校验+扰动
  T2: 解答扰动问题 → EVS筛选
  T3: Paraphrase → 构造新的问题
  T4: 解答paraphrase后的问题 → 计算reward

阶段启用控制 (通过配置):
  - enable_selfplay=False时, 只运行T4 (等同于标准GRPO)
  - enable_selfplay=True时, 运行全部T1~T4
  - enable_t2_evs, enable_t3_paraphrase 可分别控制T2和T3

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
    compute_t1_reward,
    compute_t2_reward,
    compute_t3_reward,
    compute_t4_reward,
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
        self.enable_t2_evs = sp_cfg.get("enable_t2_evs", True)
        self.enable_t3_paraphrase = sp_cfg.get("enable_t3_paraphrase", True)

        # rollout次数
        self.n1 = sp_cfg.get("n1", 4)   # T1: 模板+代码生成
        self.n2 = sp_cfg.get("n2", 5)   # 扰动次数 (非模型)
        self.n3 = sp_cfg.get("n3", 8)   # T2: 解答扰动题
        self.n4 = sp_cfg.get("n4", 4)   # T3: paraphrase
        self.n5 = sp_cfg.get("n5", 8)   # T4: 解答paraphrase题

        # loss权重
        self.w1 = sp_cfg.get("w1", 0.2)  # T1 loss权重
        self.w2 = sp_cfg.get("w2", 0.3)  # T2 loss权重
        self.w3 = sp_cfg.get("w3", 0.2)  # T3 loss权重
        self.w4 = sp_cfg.get("w4", 0.3)  # T4 loss权重

        # 扰动参数
        self.perturb_alpha = sp_cfg.get("perturb_alpha", 5)
        self.perturb_timeout = sp_cfg.get("perturb_timeout", 30)
        self.code_timeout = sp_cfg.get("code_timeout", 2.0)

        # 各阶段最大序列长度
        self.max_template_code_length = sp_cfg.get("max_template_code_length", 2048)
        self.max_solve_length = sp_cfg.get("max_solve_length", 2048)
        self.max_paraphrase_length = sp_cfg.get("max_paraphrase_length", 1024)

        mode_str = "Self-Play (T1~T4)" if self.enable_selfplay else "T4-Only (标准GRPO)"
        logger.info(f"---SELFPLAY--- 模式: {mode_str}")
        if self.enable_selfplay:
            logger.info(f"---SELFPLAY--- T2_EVS: {self.enable_t2_evs}, T3_Paraphrase: {self.enable_t3_paraphrase}")
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
            stage_batches: [t1_batch, t2_batch, t3_batch, t4_batch]
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

        stage_batches = [None, None, None, None]  # T1, T2, T3, T4

        # ====== Stage 1: 生成模板+代码 (T1) ======
        t1_batch = None
        templates = [None] * n_prompts
        python_codes = [None] * n_prompts
        verify_passed = [False] * n_prompts
        perturb_passed = [False] * n_prompts
        evs_passed_list = [False] * n_prompts

        with marked_timer("stage1_t1", timing_raw, color="blue"):
            logger.info("---STAGE1--- T1: 生成模板和代码")
            print("---STAGE1--- T1: 生成模板和代码")

            # 构造T1 prompt
            t1_prompt_batch = self.prompt_builder.build_t1_prompts(
                queries=queries,
                responses=responses_for_t1,
                max_length=self.max_template_code_length,
            )

            # 生成
            t1_batch = self._generate_for_stage(
                t1_prompt_batch, self.n1, "T1", timing_raw
            )

            # 解码并校验
            t1_responses = self._decode_responses(t1_batch)

            # 对每个原始问题, 检查其n1个rollout中是否有通过校验的
            for p_idx in range(n_prompts):
                for r_idx in range(self.n1):
                    flat_idx = p_idx * self.n1 + r_idx
                    if flat_idx >= len(t1_responses):
                        break
                    result = parse_and_verify(
                        generation=t1_responses[flat_idx],
                        query=queries[p_idx],
                        answer=answers[p_idx],
                        executor=self.executor,
                        code_timeout=self.code_timeout,
                    )
                    if result is not None:
                        templates[p_idx] = result["template"]
                        python_codes[p_idx] = result["python"]
                        verify_passed[p_idx] = True
                        break  # 一个通过即可

            passed_t1 = sum(verify_passed)
            logger.info(f"---STAGE1--- T1校验通过: {passed_t1}/{n_prompts}")
            print(f"---STAGE1--- T1校验通过: {passed_t1}/{n_prompts}")
            metrics["selfplay/t1_verify_pass_rate"] = passed_t1 / max(n_prompts, 1)

        # ====== Stage 1.5: 自动扰动 (CPU, 非模型) ======
        perturbed_data = {}  # p_idx -> list of perturbation dicts

        with marked_timer("stage1.5_perturb", timing_raw, color="blue"):
            logger.info("---STAGE1.5--- 自动扰动")
            print("---STAGE1.5--- 自动扰动")

            for p_idx in range(n_prompts):
                if not verify_passed[p_idx]:
                    continue
                perturbations = perturb_variables(
                    template=templates[p_idx],
                    python_code=python_codes[p_idx],
                    answer=answers[p_idx],
                    executor=self.executor,
                    n_perturbations=self.n2,
                    alpha_list=[self.perturb_alpha],
                    timeout_total=self.perturb_timeout,
                    code_timeout=self.code_timeout,
                )
                if perturbations:
                    perturbed_data[p_idx] = perturbations
                    perturb_passed[p_idx] = True

            passed_perturb = sum(perturb_passed)
            logger.info(f"---STAGE1.5--- 扰动通过: {passed_perturb}/{n_prompts}")
            print(f"---STAGE1.5--- 扰动通过: {passed_perturb}/{n_prompts}")
            metrics["selfplay/perturb_pass_rate"] = passed_perturb / max(n_prompts, 1)

        # ====== Stage 2: 解答扰动问题 (T2) ======
        t2_batch = None
        passed_perturbations = {}  # p_idx -> list of perturbation dicts that passed EVS

        if self.enable_t2_evs and perturbed_data:
            with marked_timer("stage2_t2", timing_raw, color="green"):
                logger.info("---STAGE2--- T2: 解答扰动问题 + EVS筛选")
                print("---STAGE2--- T2: 解答扰动问题 + EVS筛选")

                # 收集所有扰动问题
                t2_queries = []
                t2_codes = []
                t2_expected_answers = []
                t2_source_indices = []  # 记录每个扰动问题来自哪个原始问题

                for p_idx, perturbations in perturbed_data.items():
                    for pert in perturbations:
                        t2_queries.append(pert["new_query"])
                        t2_codes.append(pert["new_code"])
                        t2_expected_answers.append(pert["new_ans"])
                        t2_source_indices.append(p_idx)

                if t2_queries:
                    # 构造T2 prompt
                    t2_prompt_batch = self.prompt_builder.build_t2_prompts(
                        queries=t2_queries,
                        codes=t2_codes,
                        max_length=self.max_solve_length,
                    )

                    # 生成
                    t2_batch = self._generate_for_stage(
                        t2_prompt_batch, self.n3, "T2", timing_raw
                    )

                    # EVS筛选: 对每个扰动问题, 检查n3个回答中是否有正确的
                    t2_responses = self._decode_responses(t2_batch)

                    for q_idx in range(len(t2_queries)):
                        responses_for_q = []
                        for r_idx in range(self.n3):
                            flat_idx = q_idx * self.n3 + r_idx
                            if flat_idx < len(t2_responses):
                                responses_for_q.append(t2_responses[flat_idx])

                        passed, _ = check_evs(
                            model_responses=responses_for_q,
                            expected_answer=t2_expected_answers[q_idx],
                        )

                        if passed:
                            p_idx = t2_source_indices[q_idx]
                            if p_idx not in passed_perturbations:
                                passed_perturbations[p_idx] = []
                            # 记录通过的扰动及其相关信息
                            pert_idx_in_list = sum(1 for si in t2_source_indices[:q_idx] if si == p_idx)
                            passed_perturbations[p_idx].append({
                                "query": t2_queries[q_idx],
                                "code": t2_codes[q_idx],
                                "answer": t2_expected_answers[q_idx],
                            })
                            evs_passed_list[p_idx] = True

                    passed_evs = sum(evs_passed_list)
                    logger.info(f"---STAGE2--- EVS通过: {passed_evs}/{n_prompts}")
                    print(f"---STAGE2--- EVS通过: {passed_evs}/{n_prompts}")
                    metrics["selfplay/evs_pass_rate"] = passed_evs / max(n_prompts, 1)
        else:
            # 不启用T2_EVS: 所有扰动都算通过
            for p_idx, perturbations in perturbed_data.items():
                passed_perturbations[p_idx] = [
                    {"query": p["new_query"], "code": p["new_code"], "answer": p["new_ans"]}
                    for p in perturbations
                ]
                evs_passed_list[p_idx] = True

        # ====== Stage 3: Paraphrase (T3) ======
        t3_batch = None
        paraphrased_questions = []  # list of (question_text, expected_answer)

        if self.enable_t3_paraphrase and passed_perturbations:
            with marked_timer("stage3_t3", timing_raw, color="yellow"):
                logger.info("---STAGE3--- T3: Paraphrase")
                print("---STAGE3--- T3: Paraphrase")

                # 收集需要paraphrase的题目 (每个原始问题取第一个通过EVS的扰动)
                t3_questions = []
                t3_expected_answers = []
                t3_source_indices = []

                for p_idx, perts in passed_perturbations.items():
                    if perts:
                        t3_questions.append(perts[0]["query"])
                        t3_expected_answers.append(perts[0]["answer"])
                        t3_source_indices.append(p_idx)

                if t3_questions:
                    t3_prompt_batch = self.prompt_builder.build_t3_prompts(
                        questions=t3_questions,
                        max_length=self.max_paraphrase_length,
                    )

                    t3_batch = self._generate_for_stage(
                        t3_prompt_batch, self.n4, "T3", timing_raw
                    )

                    # 解码paraphrase结果
                    t3_responses = self._decode_responses(t3_batch)

                    # 每个原始问题取第一个非空paraphrase结果
                    for q_idx in range(len(t3_questions)):
                        for r_idx in range(self.n4):
                            flat_idx = q_idx * self.n4 + r_idx
                            if flat_idx < len(t3_responses) and t3_responses[flat_idx].strip():
                                paraphrased_questions.append((
                                    t3_responses[flat_idx].strip(),
                                    t3_expected_answers[q_idx],
                                ))
                                break

                    logger.info(f"---STAGE3--- Paraphrase完成: {len(paraphrased_questions)}个问题")
                    print(f"---STAGE3--- Paraphrase完成: {len(paraphrased_questions)}个问题")
                    metrics["selfplay/paraphrase_count"] = len(paraphrased_questions)
        else:
            # 不启用T3: 直接用扰动后的问题
            for p_idx, perts in passed_perturbations.items():
                if perts:
                    paraphrased_questions.append((
                        perts[0]["query"],
                        perts[0]["answer"],
                    ))

        # ====== Stage 4: 解答paraphrase后的问题 (T4) ======
        t4_batch = None

        if paraphrased_questions:
            with marked_timer("stage4_t4", timing_raw, color="red"):
                logger.info(f"---STAGE4--- T4: 解答{len(paraphrased_questions)}个问题")
                print(f"---STAGE4--- T4: 解答{len(paraphrased_questions)}个问题")

                t4_questions = [q for q, _ in paraphrased_questions]
                t4_expected_answers = [a for _, a in paraphrased_questions]

                t4_prompt_batch = self.prompt_builder.build_t4_prompts(
                    questions=t4_questions,
                    max_length=self.max_solve_length,
                )

                t4_batch = self._generate_for_stage(
                    t4_prompt_batch, self.n5, "T4", timing_raw
                )

                logger.info(f"---STAGE4--- T4生成完成")
                print(f"---STAGE4--- T4生成完成")

        # ====== 计算各阶段reward ======
        logger.info("---REWARD--- 开始计算各阶段reward")
        print("---REWARD--- 开始计算各阶段reward")

        # T1 reward
        if t1_batch is not None:
            t1_rewards = compute_t1_reward(
                batch_size=len(t1_batch),
                verify_passed=[verify_passed[i // self.n1] for i in range(len(t1_batch))],
                perturb_passed=[perturb_passed[i // self.n1] for i in range(len(t1_batch))],
                evs_passed=[evs_passed_list[i // self.n1] for i in range(len(t1_batch))],
            )
            # 放到token_level_scores
            seq_len = t1_batch.batch["attention_mask"].shape[1]
            t1_batch.batch["token_level_scores"] = torch.zeros(len(t1_batch), seq_len)
            t1_batch.batch["token_level_scores"][:, -1] = t1_rewards
            t1_batch.batch["token_level_rewards"] = t1_batch.batch["token_level_scores"].clone()
            metrics["selfplay/t1_avg_reward"] = t1_rewards.mean().item()

        # T4 reward (先计算, 因为T3 reward依赖T4的accuracy)
        t4_accuracies = []
        if t4_batch is not None:
            t4_responses = self._decode_responses(t4_batch)
            t4_expected = [a for _, a in paraphrased_questions]
            t4_reward_scores, t4_accuracies = compute_t4_reward(
                responses=t4_responses,
                expected_answers=t4_expected,
                group_size=self.n5,
            )
            seq_len = t4_batch.batch["attention_mask"].shape[1]
            t4_batch.batch["token_level_scores"] = torch.zeros(len(t4_batch), seq_len)
            for i in range(len(t4_batch)):
                t4_batch.batch["token_level_scores"][i, -1] = t4_reward_scores[i]
            t4_batch.batch["token_level_rewards"] = t4_batch.batch["token_level_scores"].clone()
            metrics["selfplay/t4_avg_reward"] = t4_reward_scores.mean().item()
            metrics["selfplay/t4_avg_accuracy"] = np.mean(t4_accuracies) if t4_accuracies else 0.0

        # T2 reward
        if t2_batch is not None:
            t2_responses = self._decode_responses(t2_batch)
            t2_reward_scores, group_has_correct, _ = compute_t2_reward(
                responses=t2_responses,
                expected_answers=t2_expected_answers,
                group_size=self.n3,
            )
            seq_len = t2_batch.batch["attention_mask"].shape[1]
            t2_batch.batch["token_level_scores"] = torch.zeros(len(t2_batch), seq_len)
            for i in range(len(t2_batch)):
                t2_batch.batch["token_level_scores"][i, -1] = t2_reward_scores[i]
            t2_batch.batch["token_level_rewards"] = t2_batch.batch["token_level_scores"].clone()
            metrics["selfplay/t2_avg_reward"] = t2_reward_scores.mean().item()

        # T3 reward (依赖T4 accuracy)
        if t3_batch is not None and t4_accuracies:
            t3_reward_scores = compute_t3_reward(
                t4_accuracies=t4_accuracies,
                group_size=self.n4,
            )
            seq_len = t3_batch.batch["attention_mask"].shape[1]
            t3_batch.batch["token_level_scores"] = torch.zeros(len(t3_batch), seq_len)
            for i in range(min(len(t3_batch), len(t3_reward_scores))):
                t3_batch.batch["token_level_scores"][i, -1] = t3_reward_scores[i]
            t3_batch.batch["token_level_rewards"] = t3_batch.batch["token_level_scores"].clone()
            metrics["selfplay/t3_avg_reward"] = t3_reward_scores.mean().item()

        stage_batches = [t1_batch, t2_batch, t3_batch, t4_batch]
        return stage_batches

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
                        t1_batch, t2_batch, t3_batch, t4_batch = stage_batches
                        n_rollouts_list = [self.n1, self.n3, self.n4, self.n5]
                        names = ["T1", "T2", "T3", "T4"]

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
                        # ====== 标准GRPO模式 (T4-only) ======
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
