#!/bin/bash
# TESTED: NO
# 实验目的: 用GSM8K风格的prompt测试orca_200, 对比boxed风格的差异
# 关键区别: 无system prompt, 用"#### 数字"格式
# 日期: 2026-04-05

set -x

export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAR_DIR="/home/zfs01/liyx/AdaR"
VENV_PYTHON="$ADAR_DIR/.venv/bin/python"
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

ORIGINAL_MODEL="$ADAR_DIR/models/Qwen3-4B"
TRAINED_MODEL="$ADAR_DIR/ckpt/adar_dapo_4b_orca900/global_step_28/merged_hf"
TEST_DATA="$VERL_DIR/data/selfplay/test_orca_200_gsm_style.parquet"
OUTPUT_DIR="$VERL_DIR/Experiments/20260404_qwen3_4b_dapo_orca900"

export CUDA_VISIBLE_DEVICES=${1:-7}
N_GPUS=1

cd "$VERL_DIR"

echo "---EVAL--- 生成: 原始模型 (GSM style prompt)"
$VENV_PYTHON -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    data.path="$TEST_DATA" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path="$OUTPUT_DIR/gen_original_gsm_style.parquet" \
    model.path="$ORIGINAL_MODEL" \
    +model.trust_remote_code=True \
    rollout.temperature=0.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.5 \
    rollout.enforce_eager=True \
    rollout.max_num_batched_tokens=16384 \
    rollout.max_model_len=16384 \
    +rollout.pipeline_model_parallel_size=1

echo "---EVAL--- 生成: 训练后模型 (GSM style prompt)"
$VENV_PYTHON -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$N_GPUS \
    data.path="$TEST_DATA" \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path="$OUTPUT_DIR/gen_trained_gsm_style.parquet" \
    model.path="$TRAINED_MODEL" \
    +model.trust_remote_code=True \
    rollout.temperature=0.0 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.5 \
    rollout.enforce_eager=True \
    rollout.max_num_batched_tokens=16384 \
    rollout.max_model_len=16384 \
    +rollout.pipeline_model_parallel_size=1

echo "---EVAL--- 评估正确率 (GSM style)"
$VENV_PYTHON "$SCRIPT_DIR/eval_math_acc.py" \
    --gen_parquet "$OUTPUT_DIR/gen_original_gsm_style.parquet" \
    --test_data "$VERL_DIR/data/selfplay/test_orca_200.parquet" \
    --label "original_gsm_style"

$VENV_PYTHON "$SCRIPT_DIR/eval_math_acc.py" \
    --gen_parquet "$OUTPUT_DIR/gen_trained_gsm_style.parquet" \
    --test_data "$VERL_DIR/data/selfplay/test_orca_200.parquet" \
    --label "trained_gsm_style"

echo "---EVAL--- 完成"
