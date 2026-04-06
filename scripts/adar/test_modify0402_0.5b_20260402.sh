#!/bin/bash
# TESTED: YES (2026-04-02, 3090-5 GPUs 1,2, Self-Play + DAPO + 部分奖励, 2 steps完成)
# 实验目的: 测试 Modify0402 的三项改动:
#   1. GRPO -> DAPO (token-level PG loss, 非对称clipping, filter_groups)
#   2. Stage1 部分奖励 (VA/EC/EVS/Stage4 acc)
#   3. 逐阶段 self-play 开关 (本次全部用本地模型, 不测试API)
# 主要配置: DAPO, 2xGPU, Qwen2.5-0.5B-Instruct, 200样本, enable_selfplay=True
# 日期: 2026-04-02
# 说明: 基于 test_adar_selfplay_full_0.5b_20260330.sh, 加入DAPO和新奖励

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=1,2
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
MODEL_PATH="/home/nfs04/model/Qwen2.5/Qwen2.5-0.5B-Instruct"
TRAIN_DATA="$VERL_DIR/data/selfplay/train_test_8.parquet"
CKPT_DIR="$VERL_DIR/ckpt/test_modify0402_0.5b"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 检查前置条件 ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "---ERROR--- 模型不存在: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
    echo "---ERROR--- 训练数据不存在: $TRAIN_DATA"
    exit 1
fi

# === 清理stale ray session ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true
find /tmp/ray_adar_* -maxdepth 0 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

LOG_FILE="$LOG_DIR/test_modify0402_0.5b_$(date +%Y%m%d_%H%M%S).log"

echo "---TEST--- 开始Modify0402测试 (DAPO + 部分奖励 + 逐阶段开关)..."
echo "---TEST--- 日志: $LOG_FILE"
cd "$VERL_DIR"

$CONDA_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=4 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=True \
    algorithm.filter_groups.metric=acc \
    algorithm.filter_groups.max_num_gen_batches=3 \
    reward_model.reward_manager=dapo \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='AdaR-SelfPlay-Test' \
    trainer.experiment_name='test_modify0402_dapo_0.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=100 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    adar_selfplay.enable_selfplay=True \
    adar_selfplay.enable_stage2_evs=True \
    adar_selfplay.enable_stage3_paraphrase=True \
    adar_selfplay.n1=2 \
    adar_selfplay.n2=2 \
    adar_selfplay.n3=2 \
    adar_selfplay.n4=2 \
    adar_selfplay.n5=2 \
    adar_selfplay.w1=0.2 \
    adar_selfplay.w2=0.3 \
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
    adar_selfplay.max_template_code_length=256 \
    adar_selfplay.max_solve_length=256 \
    adar_selfplay.max_paraphrase_length=256 \
    adar_selfplay.perturb_timeout=30 \
    adar_selfplay.code_timeout=3.0 \
    adar_selfplay.debug_inject_stage1=True \
    2>&1 | tee "$LOG_FILE"

echo "---TEST--- 测试完成, exit code: $?"
echo "---TEST--- 日志: $LOG_FILE"
echo ""
echo "=== 检查项 ==="
echo "1. DAPO: 确认日志中 loss_agg_mode=token-mean, clip_ratio_low=0.2, clip_ratio_high=0.28"
echo "2. Stage1奖励: 确认 STAGE1_REWARD 日志显示 VA/EC/EVS 分别计数, 奖励值非二元"
echo "3. w3=0: 确认 Stage3 loss权重为0"
echo "4. filter_groups: 确认 FILTER_GROUPS 日志存在"
