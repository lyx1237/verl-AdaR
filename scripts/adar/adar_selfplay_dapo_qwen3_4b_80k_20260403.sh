#!/bin/bash
# TESTED: NO
# 实验目的: Qwen3-4B DAPO Self-Play 训练 (Stage1->Stage2->Stage3->Stage4)
# 主要改进 (相比 0401 版):
#   1. GRPO -> DAPO: token-level PG loss, 非对称clipping (0.2/0.28), dynamic sampling
#   2. Stage1 部分奖励: VA(0.2) + EC(0.3) + EVS(0.2) + Stage4 acc(剩余)
#   3. w3=0: Stage3 暂不参与训练
#   4. Dynamic Sampling: 全阶段过滤 reward std==0 的 group (全对+全错)
# 主要配置: DAPO, 4xGPU, Qwen3-4B, 80k样本, enable_selfplay=True, 1 epoch
# 日期: 2026-04-03

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=0,2,3,5
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
TRAIN_DATA="$VERL_DIR/data/selfplay/train_selfplay_80k.parquet"
CKPT_DIR="$VERL_DIR/ckpt/adar_dapo_4b_80k"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 检查前置条件 ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "---ERROR--- 模型不存在: $MODEL_PATH"
    exit 1
fi

# === 准备数据 (如果不存在) ===
if [ ! -f "$TRAIN_DATA" ]; then
    echo "---PREP--- 准备self-play训练数据..."
    $CONDA_PYTHON "$SCRIPT_DIR/prepare_selfplay_data.py" \
        "$VERL_DIR/data/raw/orca_80k.json" \
        "$TRAIN_DATA"
    if [ $? -ne 0 ]; then
        echo "---ERROR--- 数据准备失败!"
        exit 1
    fi
fi

# === 清理stale ray session ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true
find /tmp/ray_adar_* -maxdepth 0 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

LOG_FILE="$LOG_DIR/adar_dapo_4b_80k_$(date +%Y%m%d_%H%M%S).log"

echo "---TRAIN--- 开始Qwen3-4B DAPO Self-Play训练..."
echo "---TRAIN--- 日志: $LOG_FILE"
cd "$VERL_DIR"

$CONDA_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
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
    trainer.experiment_name='qwen3_4b_dapo_80k' \
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
