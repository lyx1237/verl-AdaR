"""
多阶段Reward计算 (adar_selfplay_reward.py)

为AdaR Self-Play的4个阶段分别计算reward:
- T1: 1 if 步骤2,3,4均通过 else 0
- T2: 1 if correct else 0 (group内全错则舍弃, 回溯标记T1失败)
- T3: 1 - 4*(acc-0.5)^2, 其中acc是T4阶段该题的正确率
- T4: 1 if correct else 0

reward放到每个sequence最后一个token的位置 (token_level_scores).
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def compute_t1_reward(
    batch_size: int,
    verify_passed: list[bool],
    perturb_passed: list[bool],
    evs_passed: list[bool],
) -> torch.Tensor:
    """
    T1 Reward: 1 if 步骤2(校验), 3(扰动), 4(EVS)均通过, else 0.

    Args:
        batch_size: batch中的样本数
        verify_passed: 每个样本是否通过自动校验
        perturb_passed: 每个样本是否成功产生了扰动
        evs_passed: 每个样本的扰动是否通过了EVS

    Returns:
        (batch_size,) 的reward tensor
    """
    rewards = torch.zeros(batch_size)
    passed_count = 0
    for i in range(batch_size):
        v = verify_passed[i] if i < len(verify_passed) else False
        p = perturb_passed[i] if i < len(perturb_passed) else False
        e = evs_passed[i] if i < len(evs_passed) else False
        if v and p and e:
            rewards[i] = 1.0
            passed_count += 1

    logger.info(f"---T1_REWARD--- 通过: {passed_count}/{batch_size}")
    return rewards


def compute_t2_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[bool], list[bool]]:
    """
    T2 Reward: 1 if correct else 0.
    如果某个prompt对应的group内所有回答都错, 则标记该prompt需要回溯.

    Args:
        responses: 模型解答列表 (已展平, 长度=n_prompts*group_size)
        expected_answers: 每个prompt对应的正确答案 (长度=n_prompts)
        group_size: 每个prompt的rollout次数
        tolerance: 数值比较容差

    Returns:
        (rewards, group_has_correct, prompt_valid):
        - rewards: (total_responses,) 的reward tensor
        - group_has_correct: 每个prompt是否至少有一个正确答案
        - prompt_valid: 与group_has_correct相同, 用于决定是否回溯
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
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

    passed = sum(group_has_correct)
    logger.info(f"---T2_REWARD--- group通过: {passed}/{n_prompts}, "
                f"总正确: {int(rewards.sum())}/{total}")

    return rewards, group_has_correct, group_has_correct


def compute_t3_reward(
    t4_accuracies: list[float],
    group_size: int,
) -> torch.Tensor:
    """
    T3 Reward: 1 - 4*(acc-0.5)^2, 其中acc是T4阶段该题的正确率.
    这个reward鼓励paraphrase产生"适当有挑战性"的题目 (acc接近0.5时reward最高).

    Args:
        t4_accuracies: 每个T3 prompt对应的T4准确率 (长度=n_prompts)
        group_size: T3每个prompt的rollout次数

    Returns:
        (n_prompts * group_size,) 的reward tensor
        每个group内的所有rollout共享同一个reward
    """
    n_prompts = len(t4_accuracies)
    total = n_prompts * group_size
    rewards = torch.zeros(total)

    for p_idx in range(n_prompts):
        acc = t4_accuracies[p_idx]
        # reward = 1 - 4*(acc-0.5)^2
        # 当acc=0.5时reward=1 (最佳), acc=0或1时reward=0 (太简单或太难)
        reward = 1.0 - 4.0 * (acc - 0.5) ** 2
        reward = max(0.0, reward)  # 确保非负
        for g_idx in range(group_size):
            flat_idx = p_idx * group_size + g_idx
            if flat_idx < total:
                rewards[flat_idx] = reward

    avg_reward = rewards.mean().item()
    logger.info(f"---T3_REWARD--- 平均reward: {avg_reward:.4f}, "
                f"平均T4准确率: {np.mean(t4_accuracies):.4f}")

    return rewards


def compute_t4_reward(
    responses: list[str],
    expected_answers: list[float],
    group_size: int,
    tolerance: float = 1e-3,
) -> tuple[torch.Tensor, list[float]]:
    """
    T4 Reward: 1 if correct else 0.
    同时返回每个prompt的准确率, 用于T3的reward计算.

    Args:
        responses: 模型解答列表 (已展平)
        expected_answers: 每个prompt对应的正确答案
        group_size: 每个prompt的rollout次数
        tolerance: 数值比较容差

    Returns:
        (rewards, accuracies):
        - rewards: (total_responses,) 的reward tensor
        - accuracies: 每个prompt的正确率列表
    """
    from .auto_pipeline import extract_last_number_from_solution

    n_prompts = len(expected_answers)
    total = n_prompts * group_size
    rewards = torch.zeros(total)
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

    logger.info(f"---T4_REWARD--- 平均准确率: {np.mean(accuracies):.4f}, "
                f"总正确: {int(rewards.sum())}/{total}")

    return rewards, accuracies


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
