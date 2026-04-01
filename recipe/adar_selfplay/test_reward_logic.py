"""
手动验证 reward 计算逻辑的正确性.
用 8 道题, 手动构造每个阶段的数据, 然后与手算的期望值对比.

场景设计:
=========
8 个原始问题 (p0~p7), n1=2 (每个问题2次模板提取rollout), 共16个T1 rollout.

Stage1 parse_and_verify 结果:
  p0: rollout0 通过, rollout1 失败
  p1: rollout0 失败, rollout1 通过
  p2: rollout0 通过, rollout1 通过  (同一问题两个rollout都通过)
  p3: rollout0 失败, rollout1 失败  (全部失败)
  p4: rollout0 通过, rollout1 失败
  p5: rollout0 失败, rollout1 通过
  p6: rollout0 通过, rollout1 失败
  p7: rollout0 失败, rollout1 失败  (全部失败)

通过的rollout: (0,0),(1,1),(2,0),(2,1),(4,0),(5,1),(6,0) → k1=7

Stage 1.5 扰动 (n2=2, 每个通过rollout做2次扰动):
  (0,0): 2个扰动成功
  (1,1): 2个扰动成功
  (2,0): 2个扰动成功
  (2,1): 0个扰动成功 (perturb_variables返回空)
  (4,0): 2个扰动成功
  (5,1): 2个扰动成功
  (6,0): 2个扰动成功
  有扰动的rollout: (0,0),(1,1),(2,0),(4,0),(5,1),(6,0) → 6个, 共12个扰动问题

Stage 2 EVS (n3=3, 每个扰动问题3次解答):
  对12个扰动问题(按顺序), EVS结果:
    (0,0)的扰动0: 通过    → passed_perturbations[(0,0)]有
    (0,0)的扰动1: 失败    →
    (1,1)的扰动0: 失败
    (1,1)的扰动1: 失败    → (1,1) 全部扰动都失败!
    (2,0)的扰动0: 通过
    (2,0)的扰动1: 通过
    (4,0)的扰动0: 通过
    (4,0)的扰动1: 失败
    (5,1)的扰动0: 失败
    (5,1)的扰动1: 失败    → (5,1) 全部失败!
    (6,0)的扰动0: 通过
    (6,0)的扰动1: 通过

  passed_perturbations:
    (0,0): 1个通过
    (2,0): 2个通过
    (4,0): 1个通过
    (6,0): 2个通过

Stage1 reward (per-rollout):
  flat_idx: p_idx * n1 + r_idx
  (0,0)=idx0: verify通过, (0,0)在passed_perturbations → reward=1
  (0,1)=idx1: verify失败 → reward=0
  (1,0)=idx2: verify失败 → reward=0
  (1,1)=idx3: verify通过, (1,1)不在passed_perturbations(全失败) → reward=0
  (2,0)=idx4: verify通过, (2,0)在passed_perturbations → reward=1
  (2,1)=idx5: verify通过, 但(2,1)不在perturbed_data(扰动为空), 也不在passed_perturbations → reward=0
  (3,0)=idx6: verify失败 → reward=0
  (3,1)=idx7: verify失败 → reward=0
  (4,0)=idx8: verify通过, (4,0)在passed_perturbations → reward=1
  (4,1)=idx9: verify失败 → reward=0
  (5,0)=idx10: verify失败 → reward=0
  (5,1)=idx11: verify通过, (5,1)不在passed_perturbations → reward=0
  (6,0)=idx12: verify通过, (6,0)在passed_perturbations → reward=1
  (6,1)=idx13: verify失败 → reward=0
  (7,0)=idx14: verify失败 → reward=0
  (7,1)=idx15: verify失败 → reward=0

  期望: [1,0, 0,0, 1,0, 0,0, 1,0, 0,0, 1,0, 0,0]

Stage2 reward (n3=3, 12个扰动问题, 共36个response):
  expected_answers 依次为各扰动的 new_ans.
  我们构造response, 使得:
    扰动0(来自(0,0)): 3次中2正确1错 → has_correct=True, mask=1,1,1
    扰动1(来自(0,0)): 3次全错          → has_correct=False, mask=0,0,0
    扰动2(来自(1,1)): 3次全错          → has_correct=False, mask=0,0,0
    扰动3(来自(1,1)): 3次全错          → has_correct=False, mask=0,0,0
    扰动4(来自(2,0)): 3次中1正确       → has_correct=True, mask=1,1,1
    扰动5(来自(2,0)): 3次中3全正确     → has_correct=True, mask=1,1,1
    扰动6(来自(4,0)): 3次中1正确       → has_correct=True, mask=1,1,1
    扰动7(来自(4,0)): 3次全错          → has_correct=False, mask=0,0,0
    扰动8(来自(5,1)): 3次全错          → has_correct=False, mask=0,0,0
    扰动9(来自(5,1)): 3次全错          → has_correct=False, mask=0,0,0
    扰动10(来自(6,0)): 3次中2正确      → has_correct=True, mask=1,1,1
    扰动11(来自(6,0)): 3次中1正确      → has_correct=True, mask=1,1,1

  Stage2 reward期望 (按flat排列, 36个):
    [1,0,1, 0,0,0, 0,0,0, 0,0,0, 1,0,0, 1,1,1, 0,1,0, 0,0,0, 0,0,0, 0,0,0, 1,0,1, 0,0,1]
  Stage2 training_mask期望:
    [1,1,1, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1, 1,1,1, 0,0,0, 0,0,0, 0,0,0, 1,1,1, 1,1,1]

Stage4 reward (n5=2, 假设有3个paraphrased问题, 共6个response):
  expected_answers: [100.0, 200.0, 300.0]
  构造response:
    问题0: 2次中1正确1错 → acc=0.5, mask=1,1
    问题1: 2次全错       → acc=0.0, mask=0,0
    问题2: 2次全正确     → acc=1.0, mask=1,1

  Stage4 reward期望: [1,0, 0,0, 1,1]
  Stage4 training_mask期望: [1,1, 0,0, 1,1]
  Stage4 accuracies期望: [0.5, 0.0, 1.0]
"""

import os
import sys
import torch
import numpy as np

# 添加项目路径 (verl根目录)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from recipe.adar_selfplay.adar_selfplay_reward import (
    compute_stage1_reward,
    compute_stage2_reward,
    compute_stage3_reward,
    compute_stage4_reward,
)

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"


def assert_tensor_equal(name, actual, expected, atol=1e-6):
    """比较tensor, 打印详细信息"""
    expected_t = torch.tensor(expected, dtype=torch.float32)
    match = torch.allclose(actual, expected_t, atol=atol)
    status = PASS if match else FAIL
    print(f"  {status} {name}")
    if not match:
        print(f"    期望: {expected_t.tolist()}")
        print(f"    实际: {actual.tolist()}")
        diff_indices = (actual - expected_t).abs() > atol
        if diff_indices.any():
            for i in range(len(actual)):
                if diff_indices[i]:
                    print(f"    [idx={i}] 期望={expected_t[i].item():.4f}, 实际={actual[i].item():.4f}")
    return match


def assert_list_equal(name, actual, expected):
    match = actual == expected
    status = PASS if match else FAIL
    print(f"  {status} {name}")
    if not match:
        print(f"    期望: {expected}")
        print(f"    实际: {actual}")
    return match


def test_stage1_reward():
    """测试 Stage1 reward 的 per-rollout 逻辑"""
    print("\n" + "=" * 60)
    print("测试 Stage1 Reward (per-rollout)")
    print("=" * 60)

    n_prompts = 8
    n1 = 2

    # 通过 parse_and_verify 的 rollout
    stage1_passed_rollouts = {
        (0, 0): True,
        (1, 1): True,
        (2, 0): True,
        (2, 1): True,
        (4, 0): True,
        (5, 1): True,
        (6, 0): True,
    }

    # 通过 EVS 的扰动 (仅有部分rollout有通过的扰动)
    passed_perturbations = {
        (0, 0): [{"query": "q", "code": "c", "answer": 1}],      # 1个通过
        (2, 0): [{"query": "q", "code": "c", "answer": 1}] * 2,  # 2个通过
        (4, 0): [{"query": "q", "code": "c", "answer": 1}],      # 1个通过
        (6, 0): [{"query": "q", "code": "c", "answer": 1}] * 2,  # 2个通过
    }
    # 注意: (1,1), (2,1), (5,1) 虽然通过了verify, 但没有扰动通过EVS

    rewards = compute_stage1_reward(
        n_prompts=n_prompts,
        n1=n1,
        stage1_passed_rollouts=stage1_passed_rollouts,
        passed_perturbations=passed_perturbations,
    )

    # 手算期望:
    # idx0=(0,0): verify通过, EVS通过 → 1
    # idx1=(0,1): verify失败 → 0
    # idx2=(1,0): verify失败 → 0
    # idx3=(1,1): verify通过, EVS全失败 → 0
    # idx4=(2,0): verify通过, EVS通过 → 1
    # idx5=(2,1): verify通过, 但不在passed_perturbations → 0
    # idx6=(3,0): verify失败 → 0
    # idx7=(3,1): verify失败 → 0
    # idx8=(4,0): verify通过, EVS通过 → 1
    # idx9=(4,1): verify失败 → 0
    # idx10=(5,0): verify失败 → 0
    # idx11=(5,1): verify通过, EVS全失败 → 0
    # idx12=(6,0): verify通过, EVS通过 → 1
    # idx13=(6,1): verify失败 → 0
    # idx14=(7,0): verify失败 → 0
    # idx15=(7,1): verify失败 → 0
    expected = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

    ok = assert_tensor_equal("Stage1 rewards", rewards, expected)

    # 验证 reward=1 的个数
    reward_1_count = int(rewards.sum().item())
    expected_count = 4  # (0,0),(2,0),(4,0),(6,0)
    ok2 = assert_list_equal("Stage1 reward=1 count", reward_1_count, expected_count)

    return ok and ok2


def test_stage2_reward():
    """测试 Stage2 reward + training_mask"""
    print("\n" + "=" * 60)
    print("测试 Stage2 Reward + Training Mask")
    print("=" * 60)

    n3 = 3  # group_size
    # 12个扰动问题, 每个3次回答, 共36个response
    expected_answers = [
        10.0,   # 扰动0
        20.0,   # 扰动1
        30.0,   # 扰动2
        40.0,   # 扰动3
        50.0,   # 扰动4
        60.0,   # 扰动5
        70.0,   # 扰动6
        80.0,   # 扰动7
        90.0,   # 扰动8
        100.0,  # 扰动9
        110.0,  # 扰动10
        120.0,  # 扰动11
    ]

    # 构造36个response字符串
    # 格式: "The answer is \\boxed{X}" 表示正确, "The answer is \\boxed{0}" 表示错误
    responses = [
        # 扰动0 (ans=10): 2正确1错 → has_correct=True
        "The answer is \\boxed{10}", "The answer is \\boxed{0}", "The answer is \\boxed{10}",
        # 扰动1 (ans=20): 全错 → has_correct=False → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动2 (ans=30): 全错 → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动3 (ans=40): 全错 → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动4 (ans=50): 1正确 → True
        "The answer is \\boxed{50}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动5 (ans=60): 全正确 → True
        "The answer is \\boxed{60}", "The answer is \\boxed{60}", "The answer is \\boxed{60}",
        # 扰动6 (ans=70): 1正确 → True
        "The answer is \\boxed{0}", "The answer is \\boxed{70}", "The answer is \\boxed{0}",
        # 扰动7 (ans=80): 全错 → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动8 (ans=90): 全错 → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动9 (ans=100): 全错 → mask=0
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{0}",
        # 扰动10 (ans=110): 2正确 → True
        "The answer is \\boxed{110}", "The answer is \\boxed{0}", "The answer is \\boxed{110}",
        # 扰动11 (ans=120): 1正确 → True
        "The answer is \\boxed{0}", "The answer is \\boxed{0}", "The answer is \\boxed{120}",
    ]

    rewards, group_has_correct, training_mask = compute_stage2_reward(
        responses=responses,
        expected_answers=expected_answers,
        group_size=n3,
    )

    # 手算期望 rewards (1=正确, 0=错误):
    expected_rewards = [
        1, 0, 1,  # 扰动0
        0, 0, 0,  # 扰动1
        0, 0, 0,  # 扰动2
        0, 0, 0,  # 扰动3
        1, 0, 0,  # 扰动4
        1, 1, 1,  # 扰动5
        0, 1, 0,  # 扰动6
        0, 0, 0,  # 扰动7
        0, 0, 0,  # 扰动8
        0, 0, 0,  # 扰动9
        1, 0, 1,  # 扰动10
        0, 0, 1,  # 扰动11
    ]

    # 手算期望 training_mask (全错group → 0):
    # 全错的group: 扰动1,2,3,7,8,9
    expected_mask = [
        1, 1, 1,  # 扰动0: has_correct
        0, 0, 0,  # 扰动1: 全错
        0, 0, 0,  # 扰动2: 全错
        0, 0, 0,  # 扰动3: 全错
        1, 1, 1,  # 扰动4: has_correct
        1, 1, 1,  # 扰动5: has_correct
        1, 1, 1,  # 扰动6: has_correct
        0, 0, 0,  # 扰动7: 全错
        0, 0, 0,  # 扰动8: 全错
        0, 0, 0,  # 扰动9: 全错
        1, 1, 1,  # 扰动10: has_correct
        1, 1, 1,  # 扰动11: has_correct
    ]

    expected_group_has_correct = [
        True, False, False, False, True, True, True, False, False, False, True, True
    ]

    ok1 = assert_tensor_equal("Stage2 rewards", rewards, expected_rewards)
    ok2 = assert_tensor_equal("Stage2 training_mask", training_mask, expected_mask)
    ok3 = assert_list_equal("Stage2 group_has_correct", group_has_correct, expected_group_has_correct)

    # 统计检查
    total_correct = int(rewards.sum().item())
    expected_total_correct = 10  # 2+0+0+0+1+3+1+0+0+0+2+1=10
    ok4 = assert_list_equal("Stage2 总正确数", total_correct, expected_total_correct)

    masked_entries = int((training_mask == 0).sum().item())
    expected_masked = 6 * 3  # 6个全错group, 每个3条
    ok5 = assert_list_equal("Stage2 masked条目数", masked_entries, expected_masked)

    return all([ok1, ok2, ok3, ok4, ok5])


def test_stage4_reward():
    """测试 Stage4 reward + training_mask + accuracies"""
    print("\n" + "=" * 60)
    print("测试 Stage4 Reward + Training Mask + Accuracies")
    print("=" * 60)

    n5 = 4  # group_size
    expected_answers = [10.0, 20.0, 30.0, 40.0, 50.0]

    responses = [
        # 问题0 (ans=10): 2正确2错 → acc=0.5, mask=1
        "\\boxed{10}", "\\boxed{10}", "\\boxed{0}", "\\boxed{0}",
        # 问题1 (ans=20): 全错 → acc=0.0, mask=0
        "\\boxed{0}", "\\boxed{0}", "\\boxed{0}", "\\boxed{0}",
        # 问题2 (ans=30): 全正确 → acc=1.0, mask=1
        "\\boxed{30}", "\\boxed{30}", "\\boxed{30}", "\\boxed{30}",
        # 问题3 (ans=40): 1正确3错 → acc=0.25, mask=1
        "\\boxed{40}", "\\boxed{0}", "\\boxed{0}", "\\boxed{0}",
        # 问题4 (ans=50): 全错 → acc=0.0, mask=0
        "\\boxed{0}", "\\boxed{0}", "\\boxed{0}", "\\boxed{0}",
    ]

    rewards, accuracies, training_mask = compute_stage4_reward(
        responses=responses,
        expected_answers=expected_answers,
        group_size=n5,
    )

    expected_rewards = [
        1, 1, 0, 0,  # 问题0
        0, 0, 0, 0,  # 问题1
        1, 1, 1, 1,  # 问题2
        1, 0, 0, 0,  # 问题3
        0, 0, 0, 0,  # 问题4
    ]

    expected_mask = [
        1, 1, 1, 1,  # 问题0: has_correct
        0, 0, 0, 0,  # 问题1: 全错
        1, 1, 1, 1,  # 问题2: has_correct
        1, 1, 1, 1,  # 问题3: has_correct
        0, 0, 0, 0,  # 问题4: 全错
    ]

    expected_accuracies = [0.5, 0.0, 1.0, 0.25, 0.0]

    ok1 = assert_tensor_equal("Stage4 rewards", rewards, expected_rewards)
    ok2 = assert_tensor_equal("Stage4 training_mask", training_mask, expected_mask)
    ok3 = assert_list_equal("Stage4 accuracies", accuracies, expected_accuracies)

    masked_groups = sum(1 for acc in accuracies if acc == 0.0)
    ok4 = assert_list_equal("Stage4 masked group数", masked_groups, 2)

    return all([ok1, ok2, ok3, ok4])


def test_stage3_reward():
    """测试 Stage3 reward (依赖T4 accuracy)"""
    print("\n" + "=" * 60)
    print("测试 Stage3 Reward (基于T4 accuracy)")
    print("=" * 60)

    # 假设2个通过EVS的扰动, n4=3
    # stage3_batch有 2*3=6 个entry
    # stage4_accuracies 对应有效的paraphrase
    stage4_accuracies = [0.5, 0.0, 1.0, 0.75]

    stage3_reward_scores = compute_stage3_reward(
        stage4_accuracies=stage4_accuracies,
        group_size=3,
    )

    # 手算:
    # acc=0.5: reward = 1 - 4*(0.5-0.5)^2 = 1.0
    # acc=0.0: reward = 1 - 4*(0.0-0.5)^2 = 1-1 = 0.0
    # acc=1.0: reward = 1 - 4*(1.0-0.5)^2 = 1-1 = 0.0
    # acc=0.75: reward = 1 - 4*(0.75-0.5)^2 = 1 - 4*0.0625 = 1-0.25 = 0.75
    expected = [
        1.0, 1.0, 1.0,      # group0: acc=0.5
        0.0, 0.0, 0.0,      # group1: acc=0.0
        0.0, 0.0, 0.0,      # group2: acc=1.0
        0.75, 0.75, 0.75,   # group3: acc=0.75
    ]

    ok = assert_tensor_equal("Stage3 rewards", stage3_reward_scores, expected)
    return ok


def test_stage3_valid_map_logic():
    """
    测试 Stage3→Stage4 的 stage3_valid_map 映射逻辑.
    模拟 trainer 中 Stage3 reward 的内联计算.
    """
    print("\n" + "=" * 60)
    print("测试 Stage3→Stage4 valid_map 映射逻辑 (模拟trainer内联计算)")
    print("=" * 60)

    n4 = 3  # 每个源问题3个paraphrase rollout
    # 假设有2个通过EVS的扰动进入T3
    # stage3_batch有 2*3=6 个entry
    n_stage3_sources = 2

    # 模拟 stage3_responses: 6个, 其中一些为空
    stage3_responses = [
        "What is 2+2?",     # (q=0, r=0): 有效
        "",                  # (q=0, r=1): 空, 跳过
        "How many apples?",  # (q=0, r=2): 有效
        "Calculate 3*4",     # (q=1, r=0): 有效
        "Find the sum",      # (q=1, r=1): 有效
        "",                  # (q=1, r=2): 空, 跳过
    ]

    # 模拟构建 stage3_valid_map (和trainer逻辑一致)
    stage3_valid_map = []
    paraphrased_questions = []
    stage3_expected_answers = [100.0, 200.0]  # 每个源问题的expected answer

    for q_idx in range(n_stage3_sources):
        for r_idx in range(n4):
            flat_idx = q_idx * n4 + r_idx
            if flat_idx < len(stage3_responses) and stage3_responses[flat_idx].strip():
                paraphrased_questions.append((
                    stage3_responses[flat_idx].strip(),
                    stage3_expected_answers[q_idx],
                ))
                stage3_valid_map.append((q_idx, r_idx))

    # 期望: 4个有效paraphrase
    print(f"  有效paraphrase数: {len(paraphrased_questions)}")
    print(f"  stage3_valid_map: {stage3_valid_map}")
    ok1 = assert_list_equal("valid paraphrase数", len(paraphrased_questions), 4)
    ok2 = assert_list_equal("stage3_valid_map", stage3_valid_map, [(0, 0), (0, 2), (1, 0), (1, 1)])

    # 模拟T4 accuracy (4个paraphrase, 各自有一个accuracy)
    stage4_accuracies = [0.5, 0.8, 0.0, 0.25]

    # 按trainer逻辑计算Stage3 reward
    stage3_reward_scores = torch.zeros(6)  # stage3_batch有6个entry

    stage4_acc_by_stage3_entry = {}
    for j, (q_idx, r_idx) in enumerate(stage3_valid_map):
        if j < len(stage4_accuracies):
            stage4_acc_by_stage3_entry[(q_idx, r_idx)] = stage4_accuracies[j]

    for q_idx in range(n_stage3_sources):
        for r_idx in range(n4):
            flat_idx = q_idx * n4 + r_idx
            if flat_idx >= 6:
                break
            if (q_idx, r_idx) in stage4_acc_by_stage3_entry:
                acc = stage4_acc_by_stage3_entry[(q_idx, r_idx)]
                reward = max(0.0, 1.0 - 4.0 * (acc - 0.5) ** 2)
                stage3_reward_scores[flat_idx] = reward

    # 手算期望:
    # (0,0)=idx0: acc=0.5  → reward = 1 - 4*0 = 1.0
    # (0,1)=idx1: 空       → reward = 0.0
    # (0,2)=idx2: acc=0.8  → reward = 1 - 4*(0.3)^2 = 1-0.36 = 0.64
    # (1,0)=idx3: acc=0.0  → reward = 1 - 4*(0.5)^2 = 0.0
    # (1,1)=idx4: acc=0.25 → reward = 1 - 4*(0.25)^2 = 1-0.25 = 0.75
    # (1,2)=idx5: 空       → reward = 0.0
    expected = [1.0, 0.0, 0.64, 0.0, 0.75, 0.0]

    ok3 = assert_tensor_equal("Stage3 mapped rewards", stage3_reward_scores, expected)

    return all([ok1, ok2, ok3])


def test_training_mask_application():
    """
    测试 training_mask 在 advantage 计算后的应用效果.
    模拟 _compute_advantage_for_stage 中的逻辑.
    """
    print("\n" + "=" * 60)
    print("测试 Training Mask 应用 (模拟advantage后处理)")
    print("=" * 60)

    batch_size = 6
    seq_len = 10

    # 模拟 response_mask (假设后3个token是response)
    response_mask = torch.zeros(batch_size, seq_len)
    response_mask[:, 7:] = 1.0  # 后3个位置

    # 模拟 advantages
    advantages = torch.randn(batch_size, seq_len)
    # 记录原始值用于检查
    original_adv = advantages.clone()

    # training_mask: 样本0,1有效, 样本2,3被mask, 样本4,5有效
    training_mask_1d = torch.tensor([1, 1, 0, 0, 1, 1], dtype=torch.float32)
    training_mask = training_mask_1d.unsqueeze(1).expand(-1, seq_len)

    # 应用 (模拟trainer逻辑)
    response_mask = response_mask * training_mask
    advantages = advantages * training_mask

    # 验证: 被mask的样本的response_mask应全为0
    ok1 = assert_tensor_equal("masked样本2 response_mask", response_mask[2], [0]*seq_len)
    ok2 = assert_tensor_equal("masked样本3 response_mask", response_mask[3], [0]*seq_len)

    # 验证: 未被mask的样本的response_mask不变
    expected_rm = torch.zeros(seq_len)
    expected_rm[7:] = 1.0
    ok3 = assert_tensor_equal("有效样本0 response_mask", response_mask[0], expected_rm.tolist())
    ok4 = assert_tensor_equal("有效样本4 response_mask", response_mask[4], expected_rm.tolist())

    # 验证: 被mask的样本advantage全为0
    ok5 = assert_tensor_equal("masked样本2 advantage", advantages[2], [0]*seq_len)
    ok6 = assert_tensor_equal("masked样本3 advantage", advantages[3], [0]*seq_len)

    # 验证: 未被mask的样本advantage不变
    ok7 = assert_tensor_equal("有效样本0 advantage", advantages[0], original_adv[0].tolist())

    masked_count = int((training_mask[:, 0] == 0).sum().item())
    ok8 = assert_list_equal("masked样本数", masked_count, 2)

    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8])


def test_end_to_end_pipeline_data_flow():
    """
    端到端测试: 模拟完整pipeline的数据流动, 验证各阶段之间的数据传递.
    """
    print("\n" + "=" * 60)
    print("测试端到端Pipeline数据流")
    print("=" * 60)

    n_prompts = 4
    n1 = 2
    n2 = 2  # 每个通过的rollout做2次扰动
    n3 = 3
    n4 = 2
    n5 = 2

    # === Stage 1 ===
    # 4个问题, 各2个rollout
    stage1_passed_rollouts = {
        (0, 0): {"template": "t0", "python": "p0"},
        (1, 0): {"template": "t1", "python": "p1"},
        (1, 1): {"template": "t1b", "python": "p1b"},
        (3, 0): {"template": "t3", "python": "p3"},
    }
    # k1=4: (0,0),(1,0),(1,1),(3,0)

    # === Stage 1.5 ===
    # 模拟: (1,1)没有生成有效扰动
    perturbed_data = {
        (0, 0): [{"new_query": "q0a", "new_code": "c0a", "new_ans": 10.0},
                 {"new_query": "q0b", "new_code": "c0b", "new_ans": 11.0}],
        (1, 0): [{"new_query": "q1a", "new_code": "c1a", "new_ans": 20.0},
                 {"new_query": "q1b", "new_code": "c1b", "new_ans": 21.0}],
        # (1,1): 没有扰动
        (3, 0): [{"new_query": "q3a", "new_code": "c3a", "new_ans": 30.0},
                 {"new_query": "q3b", "new_code": "c3b", "new_ans": 31.0}],
    }
    # 6个扰动问题

    # === Stage 2 (EVS) ===
    # 模拟EVS结果
    passed_perturbations = {
        (0, 0): [{"query": "q0a", "code": "c0a", "answer": 10.0}],  # 只有扰动0通过
        (1, 0): [{"query": "q1a", "code": "c1a", "answer": 20.0},
                 {"query": "q1b", "code": "c1b", "answer": 21.0}],  # 两个都通过
        # (3,0): 全部失败
    }

    # === Stage1 Reward ===
    stage1_rewards = compute_stage1_reward(
        n_prompts=n_prompts,
        n1=n1,
        stage1_passed_rollouts=stage1_passed_rollouts,
        passed_perturbations=passed_perturbations,
    )

    # 手算:
    # (0,0)=idx0: verify通过, (0,0)在passed → 1
    # (0,1)=idx1: verify失败 → 0
    # (1,0)=idx2: verify通过, (1,0)在passed → 1
    # (1,1)=idx3: verify通过, (1,1)不在passed(没扰动) → 0
    # (2,0)=idx4: verify失败 → 0
    # (2,1)=idx5: verify失败 → 0
    # (3,0)=idx6: verify通过, (3,0)不在passed(EVS全失败) → 0
    # (3,1)=idx7: verify失败 → 0
    expected_stage1 = [1, 0, 1, 0, 0, 0, 0, 0]
    ok1 = assert_tensor_equal("E2E Stage1 rewards", stage1_rewards, expected_stage1)

    # === 收集进入T3的问题 ===
    # passed_perturbations: (0,0)有1个, (1,0)有2个 → a=3
    stage3_questions = []
    for key, perts in passed_perturbations.items():
        for pert in perts:
            stage3_questions.append(pert["query"])
    ok2 = assert_list_equal("E2E Stage3源问题数(a)", len(stage3_questions), 3)

    # === Stage3 生成 ===
    # a=3, n4=2 → stage3_batch有6个entry
    # 假设paraphrase结果:
    stage3_responses = [
        "Para q0a v1",  # (0,0): 有效
        "Para q0a v2",  # (0,1): 有效
        "",             # (1,0): 空
        "Para q1a v2",  # (1,1): 有效
        "Para q1b v1",  # (2,0): 有效
        "Para q1b v2",  # (2,1): 有效
    ]
    stage3_expected_answers = [10.0, 20.0, 21.0]

    stage3_valid_map = []
    paraphrased_questions = []
    for q_idx in range(3):
        for r_idx in range(n4):
            flat_idx = q_idx * n4 + r_idx
            if flat_idx < len(stage3_responses) and stage3_responses[flat_idx].strip():
                paraphrased_questions.append((
                    stage3_responses[flat_idx].strip(),
                    stage3_expected_answers[q_idx],
                ))
                stage3_valid_map.append((q_idx, r_idx))

    # b = 有效paraphrase数
    b = len(paraphrased_questions)
    ok3 = assert_list_equal("E2E 有效paraphrase数(b)", b, 5)
    ok4 = assert_list_equal("E2E stage3_valid_map", stage3_valid_map,
                           [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1)])

    # === Stage4 ===
    # b=5, n5=2 → 10个response
    stage4_expected = [a for _, a in paraphrased_questions]
    # [10.0, 10.0, 20.0, 21.0, 21.0]
    ok5 = assert_list_equal("E2E Stage4 expected answers",
                           stage4_expected, [10.0, 10.0, 20.0, 21.0, 21.0])

    stage4_responses = [
        "\\boxed{10}", "\\boxed{0}",   # 问题0 (ans=10): 1正确 → acc=0.5
        "\\boxed{0}", "\\boxed{0}",    # 问题1 (ans=10): 全错 → acc=0.0, mask=0
        "\\boxed{20}", "\\boxed{20}",  # 问题2 (ans=20): 全正确 → acc=1.0
        "\\boxed{21}", "\\boxed{0}",   # 问题3 (ans=21): 1正确 → acc=0.5
        "\\boxed{21}", "\\boxed{21}",  # 问题4 (ans=21): 全正确 → acc=1.0
    ]

    stage4_rewards, stage4_accuracies, stage4_mask = compute_stage4_reward(
        responses=stage4_responses,
        expected_answers=stage4_expected,
        group_size=n5,
    )

    expected_stage4_rewards = [1, 0, 0, 0, 1, 1, 1, 0, 1, 1]
    expected_stage4_mask = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
    expected_stage4_acc = [0.5, 0.0, 1.0, 0.5, 1.0]

    ok6 = assert_tensor_equal("E2E Stage4 rewards", stage4_rewards, expected_stage4_rewards)
    ok7 = assert_tensor_equal("E2E Stage4 training_mask", stage4_mask, expected_stage4_mask)
    ok8 = assert_list_equal("E2E Stage4 accuracies", stage4_accuracies, expected_stage4_acc)

    # === Stage3 Reward (内联计算) ===
    stage3_reward_scores = torch.zeros(6)  # 3*2=6
    stage4_acc_by_stage3_entry = {}
    for j, (q_idx, r_idx) in enumerate(stage3_valid_map):
        if j < len(stage4_accuracies):
            stage4_acc_by_stage3_entry[(q_idx, r_idx)] = stage4_accuracies[j]

    for q_idx in range(3):
        for r_idx in range(n4):
            flat_idx = q_idx * n4 + r_idx
            if flat_idx >= 6:
                break
            if (q_idx, r_idx) in stage4_acc_by_stage3_entry:
                acc = stage4_acc_by_stage3_entry[(q_idx, r_idx)]
                reward = max(0.0, 1.0 - 4.0 * (acc - 0.5) ** 2)
                stage3_reward_scores[flat_idx] = reward

    # 手算:
    # (0,0)=idx0: acc=0.5 → 1-0 = 1.0
    # (0,1)=idx1: acc=0.0 → 1-1 = 0.0
    # (1,0)=idx2: 空      → 0.0
    # (1,1)=idx3: acc=1.0 → 1-1 = 0.0
    # (2,0)=idx4: acc=0.5 → 1-0 = 1.0
    # (2,1)=idx5: acc=1.0 → 1-1 = 0.0
    expected_stage3 = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ok9 = assert_tensor_equal("E2E Stage3 rewards", stage3_reward_scores, expected_stage3)

    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9])


if __name__ == "__main__":
    print("=" * 60)
    print("AdaR Self-Play Reward 逻辑验证测试")
    print("=" * 60)

    results = []
    results.append(("Stage1 Reward", test_stage1_reward()))
    results.append(("Stage2 Reward + Mask", test_stage2_reward()))
    results.append(("Stage4 Reward + Mask", test_stage4_reward()))
    results.append(("Stage3 Reward (compute_stage3_reward)", test_stage3_reward()))
    results.append(("Stage3 valid_map 映射", test_stage3_valid_map_logic()))
    results.append(("Training Mask 应用", test_training_mask_application()))
    results.append(("端到端Pipeline", test_end_to_end_pipeline_data_flow()))

    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = PASS if ok else FAIL
        print(f"  {status} {name}")
        if not ok:
            all_pass = False

    if all_pass:
        print(f"\n  \033[92m全部测试通过!\033[0m")
    else:
        print(f"\n  \033[91m存在失败的测试!\033[0m")
        sys.exit(1)
