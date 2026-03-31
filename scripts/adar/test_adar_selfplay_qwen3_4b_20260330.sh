#!/bin/bash
# TESTED: YES (2026-03-30, A6000-6 GPUs 3,4, 完整Self-Play模式, 25 steps完成, 11/25 steps触发完整T1→T2→T3→T4 pipeline)
# 实验目的: 用Qwen3-4B测试AdaR Self-Play完整4阶段pipeline (T1→T2→T3→T4)
# 主要配置: GRPO, 2xGPU (A6000-6), Qwen3-4B, 200样本, enable_selfplay=True
# 日期: 2026-03-30
# 说明: 4B模型应该能生成有效的T1模板+代码, 从而触发完整T2→T3→T4 pipeline
#       使用A6000-6的空闲GPU

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=3,4
export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
VERL_DIR="/home/zfs01/liyx/verl"
# 确保venv中的工具在PATH中
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
# 将AdaR scripts加入PYTHONPATH (供reward_func导入)
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
# 避免连接到远程或其他用户的Ray集群
unset RAY_ADDRESS
# 禁用flashinfer (避免CUDA版本兼容性问题)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_ENABLE_AOT=0

# === 路径配置 ===
MODEL_PATH="$PROJECT_DIR/models/Qwen3-4B"
TRAIN_DATA="$PROJECT_DIR/data/selfplay/train_selfplay_200.parquet"
CKPT_DIR="$PROJECT_DIR/ckpt/test_selfplay_qwen3_4b"
LOG_DIR="$PROJECT_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 检查模型是否存在 ===
if [ ! -d "$MODEL_PATH" ]; then
    echo "---ERROR--- 模型不存在: $MODEL_PATH"
    echo "请先下载 Qwen3-4B 模型"
    exit 1
fi

# === 预处理: 准备数据 (如果不存在) ===
if [ ! -f "$TRAIN_DATA" ]; then
    echo "---PREP--- 准备self-play训练数据..."
    $VENV_PYTHON "$SCRIPT_DIR/prepare_selfplay_data.py" \
        "$PROJECT_DIR/data/raw/orca_200.json" \
        "$TRAIN_DATA"

    if [ $? -ne 0 ]; then
        echo "---ERROR--- 数据准备失败!"
        exit 1
    fi
fi

# === 清理stale ray session (只清理自己拥有的) ===
find /tmp/ray -maxdepth 1 -user "$(whoami)" -exec rm -rf {} + 2>/dev/null || true

# === 完整Self-Play测试训练 (Qwen3-4B) ===
echo "---TRAIN--- 开始Qwen3-4B完整Self-Play测试训练..."
cd "$VERL_DIR"

$VENV_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='AdaR-SelfPlay-test' \
    trainer.experiment_name='test_selfplay_qwen3_4b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=50 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    adar_selfplay.enable_selfplay=True \
    adar_selfplay.enable_t2_evs=True \
    adar_selfplay.enable_t3_paraphrase=True \
    adar_selfplay.n1=2 \
    adar_selfplay.n2=2 \
    adar_selfplay.n3=2 \
    adar_selfplay.n4=2 \
    adar_selfplay.n5=2 \
    adar_selfplay.max_template_code_length=512 \
    adar_selfplay.max_solve_length=512 \
    adar_selfplay.max_paraphrase_length=512 \
    adar_selfplay.perturb_timeout=10 \
    adar_selfplay.code_timeout=2.0 \
    2>&1 | tee "$LOG_DIR/test_selfplay_qwen3_4b_$(date +%Y%m%d_%H%M%S).log"

echo "---TEST--- Qwen3-4B Self-Play测试完成, exit code: $?"
