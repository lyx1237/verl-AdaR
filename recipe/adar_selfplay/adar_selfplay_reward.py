"""
多阶段Reward计算 (adar_selfplay_reward.py)

为AdaR Self-Play的4个阶段分别计算reward:
- Stage1 (per-rollout): parse_and_verify未通过→0, 所有扰动失败→0, 至少一个扰动通过EVS→1
- Stage2: 1 if correct else 0 (group内全错则masked out, 不参与参数更新)
- Stage3: 1 - 4*(acc-0.5)^2, 其中acc是Stage4阶段该paraphrase的正确率
- Stage4: 1 if correct else 0 (group内全错则masked out, 不参与参数更新)

reward放到每个sequence最后一个token的位置 (token_level_scores).
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_stage1_reward(
    n_prompts: int,
    n1: int,
    stage1_va_passed: dict,
    stage1_ec_passed: dict,
    passed_perturbations: dict,
    stage4_acc_per_rollout: dict = None,
    r1_val: float = 0.2,
    r2_val: float = 0.3,
    r3_val: float = 0.2,
    r4_mode: int = 1,
) -> torch.Tensor:
    """
    Stage1 Reward (per-rollout): 部分奖励机制.

    对于每个rollout:
    - r1: VA(变量对齐)通过 → r1_val, 否则 0
    - r2: EC(代码执行+答案验证)通过 → r2_val, 否则 0
    - r3: EVS(至少一个扰动通过)通过 → r3_val, 否则 0
    - r4: 基于Stage4正确率, r4 = (1-r1-r2-r3) * (1-(acc-0.5)^2)
      - r4_mode=1: 过滤全错组后的平均正确率
      - r4_mode=2: 直接用所有组的总正确率
    - 总奖励 = r1 + r2 + r3 + r4

    Args:
        n_prompts: 原始问题数
        n1: 每个问题的rollout次数
        stage1_va_passed: {(p_idx, r_idx): True} VA通过的rollout
        stage1_ec_passed: {(p_idx, r_idx): True} EC通过的rollout
        passed_perturbations: {(p_idx, r_idx): [perturbation_dicts]} 通过EVS的扰动
        stage4_acc_per_rollout: {(p_idx, r_idx): {"acc": float, "acc_filtered": float}}
            每个rollout衍生的Stage4正确率 (直接版和过滤版)
        r1_val: VA通过奖励值
        r2_val: EC通过奖励值
        r3_val: EVS通过奖励值
        r4_mode: 1=过滤版(舍弃全错组), 2=直接版(所有组)

    Returns:
        (n_prompts * n1,) 的reward tensor
    """
    if stage4_acc_per_rollout is None:
        stage4_acc_per_rollout = {}

    batch_size = n_prompts * n1
    rewards = torch.zeros(batch_size)

    va_count = 0
    ec_count = 0
    evs_count = 0
    r4_count = 0

    for p_idx in range(n_prompts):
        for r_idx in range(n1):
            flat_idx = p_idx * n1 + r_idx
            if flat_idx >= batch_size:
                break
            key = (p_idx, r_idx)

            r1 = 0.0
            r2 = 0.0
            r3 = 0.0
            r4 = 0.0

            # r1: VA通过
            if key in stage1_va_passed:
                r1 = r1_val
                va_count += 1

            # r2: EC通过 (VA通过是EC通过的前提)
            if key in stage1_ec_passed:
                r2 = r2_val
                ec_count += 1

            # r3: EVS通过 (EC通过是EVS的前提)
            if key in passed_perturbations and len(passed_perturbations[key]) > 0:
                r3 = r3_val
                evs_count += 1

            # r4: Stage4正确率
            if key in stage4_acc_per_rollout:
                acc_info = stage4_acc_per_rollout[key]
                if r4_mode == 1:
                    acc = acc_info.get("acc_filtered", 0.0)
                else:
                    acc = acc_info.get("acc", 0.0)
                # r4 = (1-r1-r2-r3) * (1-(acc-0.5)^2)
                remaining = 1.0 - r1 - r2 - r3
                r4 = remaining * (1.0 - (acc - 0.5) ** 2)
                r4 = max(0.0, r4)
                r4_count += 1

            rewards[flat_idx] = r1 + r2 + r3 + r4

    # DAPO Dynamic Sampling: 按prompt分组, 过滤同一prompt下所有rollout reward相同的group
    training_mask = torch.ones(batch_size)
    filtered_count = 0
    for p_idx in range(n_prompts):
        group_rewards = []
        for r_idx in range(n1):
            flat_idx = p_idx * n1 + r_idx
            if flat_idx < batch_size:
                group_rewards.append(rewards[flat_idx].item())
        if len(group_rewards) > 1 and np.std(group_rewards) == 0:
            for r_idx in range(n1):
                flat_idx = p_idx * n1 + r_idx
                if flat_idx < batch_size:
                    training_mask[flat_idx] = 0.0
            filtered_count += 1

    logger.info(f"---STAGE1_REWARD--- VA通过: {va_count}/{batch_size}, "
                f"EC通过: {ec_count}/{batch_size}, EVS通过: {evs_count}/{batch_size}, "
                f"r4有效: {r4_count}/{batch_size}, "
                f"平均reward: {rewards.mean().item():.4f}, "
                f"dynamic_sampling过滤: {filtered_count}/{n_prompts} groups")
    return rewards, training_mask


def compute_stage2_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[bool], torch.Tensor]:
    """
    Stage2 Reward: 1 if correct else 0.
    DAPO Dynamic Sampling: 过滤reward std==0的group (全错或全对), 不参与参数更新.

    Args:
        responses: 模型解答列表 (已展平, 长度=n_prompts*group_size)
        expected_answers: 每个prompt对应的正确答案 (长度=n_prompts)
        group_size: 每个prompt的rollout次数 (n3)
        tolerance: 数值比较容差

    Returns:
        (rewards, group_has_correct, training_mask):
        - rewards: (total_responses,) 的reward tensor
        - group_has_correct: 每个扰动是否至少有一个正确答案
        - training_mask: (total_responses,) 的mask, std==0 group的所有条目为0
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
    training_mask = torch.ones(total)
    group_has_correct = []

    all_correct_count = 0
    all_wrong_count = 0

    for p_idx in range(n_prompts):
        correct_count = 0
        for g_idx in range(group_size):
            flat_idx = p_idx * group_size + g_idx
            if flat_idx >= len(responses):
                break
            extracted = extract_last_number_from_solution(responses[flat_idx])
            if extracted is not None and abs(extracted - expected_answers[p_idx]) < tolerance:
                rewards[flat_idx] = 1.0
                correct_count += 1
        has_correct = correct_count > 0
        group_has_correct.append(has_correct)
        # DAPO: 过滤全错(std=0)和全对(std=0)的group
        if correct_count == 0 or correct_count == group_size:
            for g_idx in range(group_size):
                flat_idx = p_idx * group_size + g_idx
                if flat_idx < total:
                    training_mask[flat_idx] = 0.0
            if correct_count == 0:
                all_wrong_count += 1
            else:
                all_correct_count += 1

    passed = sum(group_has_correct)
    masked_total = all_wrong_count + all_correct_count
    logger.info(f"---STAGE2_REWARD--- group通过: {passed}/{n_prompts}, "
                f"总正确: {int(rewards.sum())}/{total}, "
                f"dynamic_sampling过滤: {masked_total} groups "
                f"(全错: {all_wrong_count}, 全对: {all_correct_count})")

    return rewards, group_has_correct, training_mask


def compute_stage3_reward(
    stage4_accuracies: list[float],
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage3 Reward: 1 - 4*(acc-0.5)^2, 其中acc是Stage4阶段该题的正确率.
    鼓励paraphrase产生"适当有挑战性"的题目 (acc接近0.5时reward最高).

    注意: Stage3 同一group内所有rollout共享相同reward (来自同一Stage4 accuracy),
    因此 std 恒为 0. 但 w3 已设为 0, 不影响训练.

    Args:
        stage4_accuracies: 每个Stage3 prompt对应的Stage4准确率 (长度=n_prompts)
        group_size: Stage3每个prompt的rollout次数

    Returns:
        (rewards, training_mask):
        - rewards: (n_prompts * group_size,) 的reward tensor
        - training_mask: (n_prompts * group_size,) 全1 (Stage3不做dynamic sampling)
    """
    n_prompts = len(stage4_accuracies)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
    training_mask = torch.ones(total)

    for p_idx in range(n_prompts):
        acc = stage4_accuracies[p_idx]
        # reward = 1 - 4*(acc-0.5)^2
        # 当acc=0.5时reward=1 (最佳), acc=0或1时reward=0 (太简单或太难)
        reward = 1.0 - 4.0 * (acc - 0.5) ** 2
        reward = max(0.0, reward)  # 确保非负
        for g_idx in range(group_size):
            flat_idx = p_idx * group_size + g_idx
            if flat_idx < total:
                rewards[flat_idx] = reward

    avg_reward = rewards.mean().item()
    logger.info(f"---STAGE3_REWARD--- 平均reward: {avg_reward:.4f}, "
                f"平均Stage4准确率: {np.mean(stage4_accuracies):.4f}")

    return rewards, training_mask


def compute_stage4_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """
    Stage4 Reward: 1 if correct else 0.
    DAPO Dynamic Sampling: 过滤reward std==0的group (全错或全对), 不参与参数更新.
    同时返回每个prompt的准确率, 用于T3和T1的reward计算.

    Args:
        responses: 模型解答列表 (已展平)
        expected_answers: 每个prompt对应的正确答案
        group_size: 每个prompt的rollout次数 (n5)
        tolerance: 数值比较容差

    Returns:
        (rewards, accuracies, training_mask):
        - rewards: (total_responses,) 的reward tensor
        - accuracies: 每个prompt的正确率列表
        - training_mask: (total_responses,) 的mask, std==0 group的所有条目为0
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
    training_mask = torch.ones(total)
    accuracies = []

    all_correct_count = 0
    all_wrong_count = 0

    for p_idx in range(n_prompts):
        correct_count = 0
        for g_idx in range(group_size):
            flat_idx = p_idx * group_size + g_idx
            if flat_idx >= len(responses):
                break
            extracted = extract_last_number_from_solution(responses[flat_idx])
            if extracted is not None and abs(extracted - expected_answers[p_idx]) < tolerance:
                rewards[flat_idx] = 1.0
                correct_count += 1
        acc = correct_count / group_size
        accuracies.append(acc)
        # DAPO: 过滤全错(std=0)和全对(std=0)的group
        if correct_count == 0 or correct_count == group_size:
            for g_idx in range(group_size):
                flat_idx = p_idx * group_size + g_idx
                if flat_idx < total:
                    training_mask[flat_idx] = 0.0
            if correct_count == 0:
                all_wrong_count += 1
            else:
                all_correct_count += 1

    masked_total = all_wrong_count + all_correct_count
    logger.info(f"---STAGE4_REWARD--- 平均准确率: {np.mean(accuracies):.4f}, "
                f"总正确: {int(rewards.sum())}/{total}, "
                f"dynamic_sampling过滤: {masked_total} groups "
                f"(全错: {all_wrong_count}, 全对: {all_correct_count})")

    return rewards, accuracies, training_mask


def place_reward_on_last_token(
    reward_scores: torch.Tensor,
    response_length: torch.Tensor,
    seq_length: int,
) -> torch.Tensor:
    """
    将scalar reward放到每个sequence的最后一个response token位置.
    其他位置为0.

    Args:
        reward_scores: (batch_size,) 的reward值
        response_length: (batch_size,) 每个sequence的response长度
        seq_length: 总sequence长度 (prompt+response)

    Returns:
        (batch_size, seq_length) 的token_level_scores
    """
    batch_size = reward_scores.shape[0]
    token_level_scores = torch.zeros(batch_size, seq_length)
    for i in range(batch_size):
        # reward放在最后一个token位置
        last_token_pos = seq_length - 1
        token_level_scores[i, last_token_pos] = reward_scores[i]
    return token_level_scores
