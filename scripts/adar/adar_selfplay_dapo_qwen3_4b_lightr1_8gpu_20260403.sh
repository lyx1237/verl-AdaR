#!/bin/bash
# TESTED: NO
# 实验目的: Qwen3-4B DAPO Self-Play 训练 (Light-R1-SFTData)
# 数据: qihoo360/Light-R1-SFTData (SFT数据转self-play格式)
# 基于 adar_selfplay_dapo_qwen3_0.6b_lightr1_20260403.sh 修改:
#   1. 模型: Qwen3-0.6B -> Qwen3-4B
#   2. GPU: 8卡(A6000-5, GPU0-7)
#   3. 环境: AdaR/.venv (lyx-adar) 替代 conda lyx-verl
#   4. batch size/lr等参数适配4B模型和8卡
#   5. param_offload/optimizer_offload关闭 (8卡显存充裕)
# 日期: 2026-04-03

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=4,5,6,7
export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
ADAR_DIR="/home/zfs01/liyx/AdaR"
VENV_PYTHON="$ADAR_DIR/.venv/bin/python"
export PATH="$ADAR_DIR/.venv/bin:$PATH"
export PYTHONPATH="$ADAR_DIR/scripts:$PYTHONPATH"
unset RAY_ADDRESS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=0

# === 路径配置 ===
MODEL_PATH="$ADAR_DIR/models/Qwen3-4B"
RAW_DATA_DIR="$VERL_DIR/data/raw/Light-R1-SFTData"
TRAIN_DATA="$VERL_DIR/data/selfplay/train_lightr1_sft.parquet"
CKPT_DIR="$ADAR_DIR/ckpt/adar_dapo_4b_lightr1_8gpu"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 检查前置条件 ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "---ERROR--- 模型不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -d "$RAW_DATA_DIR" ]; then
    echo "---ERROR--- 数据集不存在: $RAW_DATA_DIR"
    exit 1
fi

# === 准备数据 (如果不存在) ===
if [ ! -f "$TRAIN_DATA" ]; then
    echo "---PREP--- 准备Light-R1-SFTData训练数据..."
    $VENV_PYTHON "$SCRIPT_DIR/prepare_lightr1_sft_data.py" \
        "$RAW_DATA_DIR" \
        "$TRAIN_DATA"
    if [ $? -ne 0 ]; then
        echo "---ERROR--- 数据准备失败!"
        exit 1
    fi
fi

# === 清理stale ray session ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true
find /tmp/ray_adar_* -maxdepth 0 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

LOG_FILE="$LOG_DIR/adar_dapo_4b_lightr1_8gpu_$(date +%Y%m%d_%H%M%S).log"

echo "---TRAIN--- 开始Qwen3-4B DAPO Self-Play训练 (8GPU, Light-R1-SFTData)..."
echo "---TRAIN--- 日志: $LOG_FILE"
cd "$VERL_DIR"

$VENV_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode="token-mean" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=10 \
    reward_model.reward_manager=dapo \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='AdaR-SelfPlay' \
    trainer.experiment_name='qwen3_4b_dapo_lightr1_8gpu' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=50 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    adar_selfplay.enable_selfplay=True \
    adar_selfplay.enable_stage2_evs=True \
    adar_selfplay.enable_stage3_paraphrase=True \
    adar_selfplay.n1=4 \
    adar_selfplay.n2=5 \
    adar_selfplay.n3=2 \
    adar_selfplay.n4=4 \
    adar_selfplay.n5=4 \
    adar_selfplay.w1=0.2 \
    adar_selfplay.w2=0.0 \
    adar_selfplay.w3=0.0 \
    adar_selfplay.w4=0.3 \
    adar_selfplay.stage1_r1=0.2 \
    adar_selfplay.stage1_r2=0.3 \
    adar_selfplay.stage1_r3=0.2 \
    adar_selfplay.stage1_r4_mode=1 \
    adar_selfplay.selfplay_stage1=True \
    adar_selfplay.selfplay_stage2=True \
    adar_selfplay.selfplay_stage3=True \
    adar_selfplay.selfplay_stage4=True \
    adar_selfplay.max_template_code_length=1024 \
    adar_selfplay.max_solve_length=1024 \
    adar_selfplay.max_paraphrase_length=1024 \
    adar_selfplay.perturb_timeout=30 \
    adar_selfplay.code_timeout=3.0 \
    2>&1 | tee "$LOG_FILE"

echo "---TRAIN--- 训练完成, exit code: $?"
echo "---TRAIN--- 日志: $LOG_FILE"
