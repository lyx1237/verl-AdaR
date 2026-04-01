#!/bin/bash
# TESTED: YES (2026-04-01, 3090-5 4xGPU, debug_inject_stage1=True, step1完成, step2因batch不整除4卡OOM)
# 实验目的: 用Qwen3-4B真实跑完整pipeline, 手动验算reward正确性
# 主要配置: GRPO, 2xGPU (3090-5 GPU0,1), Qwen3-4B, 8样本, enable_selfplay=True, 无debug注入
# 日期: 2026-04-01
# 说明: 4B模型应能生成有效template+code, 触发完整Stage1→Stage2→Stage3→Stage4 pipeline

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_PYTHON="conda run --no-capture-output -n lyx-verl python"
unset RAY_ADDRESS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=0

# === 路径配置 ===
MODEL_PATH="$VERL_DIR/models/Qwen3-4B"
TRAIN_DATA="$VERL_DIR/data/selfplay/train_test_8.parquet"
CKPT_DIR="$VERL_DIR/ckpt/test_reward_verify_4b"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 检查前置条件 ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "---ERROR--- 模型不存在: $MODEL_PATH"
    exit 1
fi
if [ ! -f "$TRAIN_DATA" ]; then
    echo "---ERROR--- 数据不存在: $TRAIN_DATA"
    exit 1
fi

echo "---GPU--- GPU状态:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

# === 清理 ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true
find /tmp/ray_adar_* -maxdepth 0 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

LOG_FILE="$LOG_DIR/test_reward_verify_4b_$(date +%Y%m%d_%H%M%S).log"

echo "---TRAIN--- 开始Qwen3-4B reward验证训练 (2步)..."
echo "---TRAIN--- 日志: $LOG_FILE"
cd "$VERL_DIR"

$CONDA_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=384 \
    data.max_response_length=384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='AdaR-SelfPlay-reward-verify' \
    trainer.experiment_name='reward_verify_4b_3090' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.total_epochs=2 \
    adar_selfplay.enable_selfplay=True \
    adar_selfplay.enable_stage3_paraphrase=True \
    adar_selfplay.n1=2 \
    adar_selfplay.n2=2 \
    adar_selfplay.n3=2 \
    adar_selfplay.n4=2 \
    adar_selfplay.n5=2 \
    adar_selfplay.max_template_code_length=384 \
    adar_selfplay.max_solve_length=384 \
    adar_selfplay.max_paraphrase_length=384 \
    adar_selfplay.perturb_timeout=10 \
    adar_selfplay.code_timeout=2.0 \
    adar_selfplay.debug_inject_stage1=True \
    2>&1 | tee "$LOG_FILE"

echo "---TEST--- Qwen3-4B reward验证完成, exit code: $?"
echo "---TEST--- 日志: $LOG_FILE"
