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
    stage1_passed_rollouts: dict,
    passed_perturbations: dict,
) -> torch.Tensor:
    """
    Stage1 Reward (per-rollout):
    - parse_and_verify 未通过 → 0
    - parse_and_verify 通过但所有扰动均未通过EVS → 0
    - parse_and_verify 通过且至少一个扰动通过EVS → 1

    Args:
        n_prompts: 原始问题数
        n1: 每个问题的rollout次数
        stage1_passed_rollouts: {(p_idx, r_idx): True} 通过parse_and_verify的rollout
        passed_perturbations: {(p_idx, r_idx): [perturbation_dicts]} 通过EVS的扰动

    Returns:
        (n_prompts * n1,) 的reward tensor
    """
    batch_size = n_prompts * n1
    rewards = torch.zeros(batch_size)
    passed_count = 0
    for p_idx in range(n_prompts):
        for r_idx in range(n1):
            flat_idx = p_idx * n1 + r_idx
            if flat_idx >= batch_size:
                break
            key = (p_idx, r_idx)
            if key in stage1_passed_rollouts:
                # parse_and_verify通过, 检查是否有扰动通过EVS
                if key in passed_perturbations and len(passed_perturbations[key]) > 0:
                    rewards[flat_idx] = 1.0
                    passed_count += 1
                # else: 所有扰动失败 → 0
            # else: parse_and_verify未通过 → 0

    logger.info(f"---STAGE1_REWARD--- 通过: {passed_count}/{batch_size} "
                f"(verify通过: {len(stage1_passed_rollouts)}, EVS有效: {len(passed_perturbations)})")
    return rewards


def compute_stage2_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[bool], torch.Tensor]:
    """
    Stage2 Reward: 1 if correct else 0.
    如果某个扰动对应的group内n3次尝试全部失败, 则该扰动被视为失败扰动,
    其所有尝试都不参与参数更新 (training_mask=0).

    Args:
        responses: 模型解答列表 (已展平, 长度=n_prompts*group_size)
        expected_answers: 每个prompt对应的正确答案 (长度=n_prompts)
        group_size: 每个prompt的rollout次数 (n3)
        tolerance: 数值比较容差

    Returns:
        (rewards, group_has_correct, training_mask):
        - rewards: (total_responses,) 的reward tensor
        - group_has_correct: 每个扰动是否至少有一个正确答案
        - training_mask: (total_responses,) 的mask, 全错group的所有条目为0
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
    training_mask = torch.ones(total)
    group_has_correct = []

    for p_idx in range(n_prompts):
        has_correct = False
        for g_idx in range(group_size):
            flat_idx = p_idx * group_size + g_idx
            if flat_idx >= len(responses):
                break
            extracted = extract_last_number_from_solution(responses[flat_idx])
            if extracted is not None and abs(extracted - expected_answers[p_idx]) < tolerance:
                rewards[flat_idx] = 1.0
                has_correct = True
        group_has_correct.append(has_correct)
        # 全错的group: 不参与参数更新
        if not has_correct:
            for g_idx in range(group_size):
                flat_idx = p_idx * group_size + g_idx
                if flat_idx < total:
                    training_mask[flat_idx] = 0.0

    passed = sum(group_has_correct)
    masked_out = sum(1 for x in group_has_correct if not x)
    logger.info(f"---STAGE2_REWARD--- group通过: {passed}/{n_prompts}, "
                f"总正确: {int(rewards.sum())}/{total}, "
                f"masked_out: {masked_out * group_size}条目")

    return rewards, group_has_correct, training_mask


def compute_stage3_reward(
    stage4_accuracies: list[float],
    group_size: int,
) -> torch.Tensor:
    """
    Stage3 Reward: 1 - 4*(acc-0.5)^2, 其中acc是Stage4阶段该题的正确率.
    鼓励paraphrase产生"适当有挑战性"的题目 (acc接近0.5时reward最高).

    Args:
        stage4_accuracies: 每个Stage3 prompt对应的Stage4准确率 (长度=n_prompts)
        group_size: Stage3每个prompt的rollout次数

    Returns:
        (n_prompts * group_size,) 的reward tensor
        每个group内的所有rollout共享同一个reward
    """
    n_prompts = len(stage4_accuracies)
    total = n_prompts * group_size
    rewards = torch.zeros(total)

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

    return rewards


def compute_stage4_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[float], torch.Tensor]:
    """
    Stage4 Reward: 1 if correct else 0.
    如果某个问题的n5次回答全部错误, 则该问题的所有回答不参与参数更新 (training_mask=0).
    同时返回每个prompt的准确率, 用于T3的reward计算.

    Args:
        responses: 模型解答列表 (已展平)
        expected_answers: 每个prompt对应的正确答案
        group_size: 每个prompt的rollout次数 (n5)
        tolerance: 数值比较容差

    Returns:
        (rewards, accuracies, training_mask):
        - rewards: (total_responses,) 的reward tensor
        - accuracies: 每个prompt的正确率列表
        - training_mask: (total_responses,) 的mask, 全错group的所有条目为0
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
    training_mask = torch.ones(total)
    accuracies = []

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
        # 全错的group: 不参与参数更新
        if correct_count == 0:
            for g_idx in range(group_size):
                flat_idx = p_idx * group_size + g_idx
                if flat_idx < total:
                    training_mask[flat_idx] = 0.0

    masked_out = sum(1 for acc in accuracies if acc == 0.0)
    logger.info(f"---STAGE4_REWARD--- 平均准确率: {np.mean(accuracies):.4f}, "
                f"总正确: {int(rewards.sum())}/{total}, "
                f"masked_out: {masked_out * group_size}条目")

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
