"""
轨迹展平模块：将 WebArena agent loop 产出的多步轨迹展平为独立样本。

输入: DataProto batch，每条轨迹在 non_tensor_batch["per_step_snapshots"] 中包含
      per-step 快照列表 [{prompt_ids, action_ids, action_mask}, ...]

输出: 展平后的 DataProto batch，每个 step 是独立样本：
      - prompts: 该步的完整上下文 token ids（含未压缩 HTML）
      - responses: 该步的 action token ids
      - response_mask: action_mask（thinking=0, answer=1）
      - attention_mask: 全 1
      - rm_scores: 同一轨迹的所有步共享同一个 trajectory-level reward
      - uid (non_tensor_batch): 同一 task 的所有轨迹的所有步归为同一 group

设计要点：
- 展平后的 batch 可直接送入标准 GRPO 训练，不需要修改 verl 的 core_algos 或 trainer
- uid 按原始 task（prompt）分组，使 GRPO advantage 在同一 task 的多条轨迹间计算
- 支持序列 padding 到统一长度（左 padding prompt，右 padding response）
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.protocol import DataProtoConfig

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def flatten_trajectories(
    batch: DataProto,
    max_prompt_length: Optional[int] = None,
    max_response_length: Optional[int] = None,
) -> DataProto:
    """
    将含 per_step_snapshots 的轨迹 batch 展平为独立样本 batch。

    Args:
        batch: rollout 产出的 DataProto，包含：
            - batch["rm_scores"]: (bs, response_length) 轨迹级 reward
            - non_tensor_batch["per_step_snapshots"]: 每条轨迹的 per-step 快照列表
            - non_tensor_batch["uid"]: 原始 uid（同一 task 的多条轨迹共享）
        max_prompt_length: prompt 最大长度（左 padding），None 则取 batch 中最大值
        max_response_length: response 最大长度（右 padding），None 则取 batch 中最大值

    Returns:
        展平后的 DataProto，每个 step 是独立样本
    """
    snapshots_arr = batch.non_tensor_batch["per_step_snapshots"]
    uid_arr = batch.non_tensor_batch["uid"]
    bs = len(snapshots_arr)

    # --- 收集所有展平样本 ---
    flat_samples = []  # list of (prompt_ids, action_ids, action_mask, reward, uid)

    for i in range(bs):
        snapshots = snapshots_arr[i]
        if snapshots is None or len(snapshots) == 0:
            logger.warning(f"---FLATTEN--- trajectory {i} has no snapshots, skipping")
            continue

        # 提取该轨迹的 reward（rm_scores 中最后一个 valid token 的值）
        if "rm_scores" in batch.batch.keys():
            rm_scores_i = batch.batch["rm_scores"][i]  # (response_length,)
            # reward 存在最后一个非零位置
            nonzero_mask = rm_scores_i.nonzero(as_tuple=False)
            if len(nonzero_mask) > 0:
                reward = rm_scores_i[nonzero_mask[-1]].item()
            else:
                reward = 0.0
        else:
            reward = 0.0

        uid = uid_arr[i]

        for step_data in snapshots:
            prompt_ids = step_data["prompt_ids"]
            action_ids = step_data["action_ids"]
            action_mask = step_data["action_mask"]

            if len(action_ids) == 0:
                logger.warning(
                    f"---FLATTEN--- trajectory {i} has empty action_ids in a step, skipping"
                )
                continue

            flat_samples.append({
                "prompt_ids": prompt_ids,
                "action_ids": action_ids,
                "action_mask": action_mask,
                "reward": reward,
                "uid": uid,
            })

    if len(flat_samples) == 0:
        logger.error("---FLATTEN--- no valid samples after flattening!")
        return batch

    n_flat = len(flat_samples)

    # --- 确定 padding 长度 ---
    if max_prompt_length is None:
        max_prompt_length = max(len(s["prompt_ids"]) for s in flat_samples)
    if max_response_length is None:
        max_response_length = max(len(s["action_ids"]) for s in flat_samples)

    logger.info(
        f"---FLATTEN--- {bs} trajectories -> {n_flat} flat samples, "
        f"max_prompt_len={max_prompt_length}, max_response_len={max_response_length}"
    )

    # --- 构建 padded tensors ---
    # prompts: 左 padding with 0
    prompts = torch.zeros(n_flat, max_prompt_length, dtype=torch.long)
    # responses: 右 padding with 0
    responses = torch.zeros(n_flat, max_response_length, dtype=torch.long)
    # response_mask: action_mask（thinking=0, answer=1），padding 部分为 0
    response_mask = torch.zeros(n_flat, max_response_length, dtype=torch.long)
    # attention_mask: 覆盖 prompt + response 的有效部分
    total_length = max_prompt_length + max_response_length
    attention_mask = torch.zeros(n_flat, total_length, dtype=torch.long)
    # rm_scores: reward 放在 response 的最后一个有效 token 位置
    rm_scores = torch.zeros(n_flat, max_response_length, dtype=torch.float32)
    # input_ids: prompt + response 拼接
    input_ids = torch.zeros(n_flat, total_length, dtype=torch.long)

    uid_list = []

    for idx, sample in enumerate(flat_samples):
        p_ids = sample["prompt_ids"]
        a_ids = sample["action_ids"]
        a_mask = sample["action_mask"]

        # 截断
        p_ids = p_ids[-max_prompt_length:]  # 保留尾部（左截断）
        a_ids = a_ids[:max_response_length]  # 保留头部（右截断）
        a_mask = a_mask[:max_response_length]

        p_len = len(p_ids)
        a_len = len(a_ids)

        # prompts: 左 padding
        prompts[idx, max_prompt_length - p_len:] = torch.tensor(p_ids, dtype=torch.long)

        # responses: 右 padding
        responses[idx, :a_len] = torch.tensor(a_ids, dtype=torch.long)

        # response_mask
        response_mask[idx, :a_len] = torch.tensor(a_mask, dtype=torch.long)

        # attention_mask: prompt 部分（左 padding）+ response 部分
        attention_mask[idx, max_prompt_length - p_len:max_prompt_length + a_len] = 1

        # rm_scores: reward 在最后一个有效 response token 位置
        if a_len > 0:
            rm_scores[idx, a_len - 1] = sample["reward"]

        # input_ids: prompt + response 拼接
        input_ids[idx, max_prompt_length - p_len:max_prompt_length] = torch.tensor(
            p_ids, dtype=torch.long
        )
        input_ids[idx, max_prompt_length:max_prompt_length + a_len] = torch.tensor(
            a_ids, dtype=torch.long
        )

        uid_list.append(sample["uid"])

    # --- 构建 DataProto ---
    tensor_batch = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts,
            "responses": responses,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rm_scores": rm_scores,
        },
        batch_size=n_flat,
    )

    non_tensor_batch = {
        "uid": np.array(uid_list, dtype=object),
    }

    # 保留原始 batch 中其他需要的 non_tensor 字段（如 data_source, reward_model 等）
    # 通过取每个展平样本对应的原始轨迹的字段来填充
    # 重新构建映射：flat_sample_idx -> original_trajectory_idx
    flat_to_orig = []
    for i in range(bs):
        snapshots = snapshots_arr[i]
        if snapshots is None or len(snapshots) == 0:
            continue
        for step_data in snapshots:
            if len(step_data.get("action_ids", [])) > 0:
                flat_to_orig.append(i)

    # 复制需要保留的 non_tensor 字段
    keys_to_copy = ["data_source", "reward_model", "extra_info", "__num_turns__"]
    for key in keys_to_copy:
        if key in batch.non_tensor_batch:
            orig_arr = batch.non_tensor_batch[key]
            new_arr = np.empty(n_flat, dtype=object)
            for flat_idx, orig_idx in enumerate(flat_to_orig):
                new_arr[flat_idx] = orig_arr[orig_idx]
            non_tensor_batch[key] = new_arr

    meta_info = batch.meta_info.copy() if batch.meta_info else {}
    # 展平后 batch size 不可预测，必须启用 auto_padding 以确保能被 DP size 整除
    meta_info[DataProtoConfig.auto_padding_key] = True

    result = DataProto(
        batch=tensor_batch,
        non_tensor_batch=non_tensor_batch,
        meta_info=meta_info,
    )

    logger.info(
        f"---FLATTEN--- output batch: {n_flat} samples, "
        f"tensor shape: input_ids={input_ids.shape}, "
        f"prompts={prompts.shape}, responses={responses.shape}"
    )
    return result
