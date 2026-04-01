#!/bin/bash
# TESTED: YES (2026-03-30, A6000-4 GPUs 0,1, 完整Self-Play模式, 24 steps完成)
# 实验目的: 测试AdaR Self-Play完整4阶段pipeline (Stage1→Stage2→Stage3→Stage4)
# 主要配置: GRPO, 2xGPU, Qwen2.5-0.5B-Instruct, 200样本, enable_selfplay=True
# 日期: 2026-03-30
# 说明: 在Stage4-only测试通过的基础上, 启用完整self-play pipeline
#       减小rollout次数以加速测试 (n1=2, n2=2, n3=2, n4=2, n5=2)

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_PYTHON="conda run --no-capture-output -n lyx-verl python"
# 避免连接到远程或其他用户的Ray集群
unset RAY_ADDRESS
# 禁用flashinfer (避免CUDA版本兼容性问题)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=0

# === 路径配置 ===
MODEL_PATH="/home/nfs04/model/Qwen2.5/Qwen2.5-0.5B-Instruct"
TRAIN_DATA="$VERL_DIR/data/selfplay/train_selfplay_200.parquet"
CKPT_DIR="$VERL_DIR/ckpt/test_selfplay_full_0.5b"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 预处理: 准备数据 (如果不存在) ===
if [ ! -f "$TRAIN_DATA" ]; then
    echo "---PREP--- 准备self-play训练数据..."
    $CONDA_PYTHON "$SCRIPT_DIR/prepare_selfplay_data.py" \
        "$VERL_DIR/data/raw/orca_200.json" \
        "$TRAIN_DATA"

    if [ $? -ne 0 ]; then
        echo "---ERROR--- 数据准备失败!"
        exit 1
    fi
fi

# === 清理stale ray session (只清理自己拥有的) ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

# === 完整Self-Play测试训练 ===
echo "---TRAIN--- 开始完整Self-Play测试训练..."
cd "$VERL_DIR"

$CONDA_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=256 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=64 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='AdaR-SelfPlay-test' \
    trainer.experiment_name='test_selfplay_full_0.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=50 \
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
    adar_selfplay.max_template_code_length=256 \
    adar_selfplay.max_solve_length=256 \
    adar_selfplay.max_paraphrase_length=256 \
    adar_selfplay.perturb_timeout=10 \
    adar_selfplay.code_timeout=2.0 \
    2>&1 | tee "$LOG_DIR/test_selfplay_full_$(date +%Y%m%d_%H%M%S).log"

echo "---TEST--- 完整Self-Play测试完成, exit code: $?"
