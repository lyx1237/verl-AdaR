#!/bin/bash
# 用<answer></answer>标签风格测试orca_200
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
TEST_DATA="$VERL_DIR/data/selfplay/test_orca_200_answer_tag.parquet"
OUTPUT_DIR="$VERL_DIR/Experiments/20260404_qwen3_4b_dapo_orca900"

export CUDA_VISIBLE_DEVICES=${1:-7}
cd "$VERL_DIR"

echo "---EVAL--- 生成: 原始模型 (answer tag)"
$VENV_PYTHON -m verl.trainer.main_generation \
    trainer.nnodes=1 trainer.n_gpus_per_node=1 \
    data.path="$TEST_DATA" data.prompt_key=prompt data.n_samples=1 \
    data.output_path="$OUTPUT_DIR/gen_original_answer_tag.parquet" \
    model.path="$ORIGINAL_MODEL" +model.trust_remote_code=True \
    rollout.temperature=0.0 rollout.prompt_length=1024 rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 rollout.gpu_memory_utilization=0.5 \
    rollout.enforce_eager=True rollout.max_num_batched_tokens=16384 rollout.max_model_len=16384 \
    +rollout.pipeline_model_parallel_size=1

echo "---EVAL--- 生成: 训练后模型 (answer tag)"
$VENV_PYTHON -m verl.trainer.main_generation \
    trainer.nnodes=1 trainer.n_gpus_per_node=1 \
    data.path="$TEST_DATA" data.prompt_key=prompt data.n_samples=1 \
    data.output_path="$OUTPUT_DIR/gen_trained_answer_tag.parquet" \
    model.path="$TRAINED_MODEL" +model.trust_remote_code=True \
    rollout.temperature=0.0 rollout.prompt_length=1024 rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 rollout.gpu_memory_utilization=0.5 \
    rollout.enforce_eager=True rollout.max_num_batched_tokens=16384 rollout.max_model_len=16384 \
    +rollout.pipeline_model_parallel_size=1

echo "---EVAL--- 评估正确率 (answer tag)"
$VENV_PYTHON "$SCRIPT_DIR/eval_math_acc.py" \
    --gen_parquet "$OUTPUT_DIR/gen_original_answer_tag.parquet" \
    --test_data "$VERL_DIR/data/selfplay/test_orca_200.parquet" \
    --label "original_answer_tag"

$VENV_PYTHON "$SCRIPT_DIR/eval_math_acc.py" \
    --gen_parquet "$OUTPUT_DIR/gen_trained_answer_tag.parquet" \
    --test_data "$VERL_DIR/data/selfplay/test_orca_200.parquet" \
    --label "trained_answer_tag"

echo "---EVAL--- 完成"
