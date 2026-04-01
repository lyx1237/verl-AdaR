#!/bin/bash
# TESTED: YES (2026-03-30, A6000-4 GPUs 4,5, Stage4-only模式, 11 steps完成)
# 实验目的: 测试AdaR Self-Play训练流程能否跑通 (Stage4-only模式, 即标准GRPO)
# 主要配置: GRPO, 2xGPU, Qwen2.5-0.5B-Instruct, 200样本, Stage4-only模式
# 日期: 2026-03-30
# 说明: 先测试Stage4-only模式 (enable_selfplay=False), 确认基本流程正常后
#       再测试完整self-play模式

set -x

# === 环境配置 ===
export CUDA_VISIBLE_DEVICES=4,5
export CUDA_HOME=/home/nfs05/cuda_tools/cuda-12.1
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONDA_PYTHON="conda run --no-capture-output -n lyx-verl python"
# 避免连接到远程或其他用户的Ray集群
unset RAY_ADDRESS

# === 路径配置 ===
MODEL_PATH="/home/nfs04/model/Qwen2.5/Qwen2.5-0.5B-Instruct"
TRAIN_DATA="$VERL_DIR/data/selfplay/train_selfplay_200.parquet"
CKPT_DIR="$VERL_DIR/ckpt/test_selfplay_0.5b"
LOG_DIR="$VERL_DIR/logs"

mkdir -p "$CKPT_DIR" "$LOG_DIR"

# === 预处理: 准备数据 ===
echo "---PREP--- 准备self-play训练数据..."
$CONDA_PYTHON "$SCRIPT_DIR/prepare_selfplay_data.py" \
    "$VERL_DIR/data/raw/orca_200.json" \
    "$TRAIN_DATA"

if [ $? -ne 0 ]; then
    echo "---ERROR--- 数据准备失败!"
    exit 1
fi

# === 预处理: 验证数据 ===
$CONDA_PYTHON -c "
import pandas as pd
df = pd.read_parquet('$TRAIN_DATA')
print(f'---DATA--- Self-Play训练数据加载成功')
print(f'---DATA--- 样本数: {len(df)}')
print(f'---DATA--- 列: {list(df.columns)}')
print(f'---DATA--- Sample extra_info keys: {list(df[\"extra_info\"].iloc[0].keys())}')
"

if [ $? -ne 0 ]; then
    echo "---ERROR--- 数据验证失败!"
    exit 1
fi

# === 验证模块导入 ===
$CONDA_PYTHON -c "
import sys
sys.path.insert(0, '$VERL_DIR')
from recipe.adar_selfplay.auto_pipeline import SafeExecutor
from recipe.adar_selfplay.prompt_builder import PromptBuilder
from recipe.adar_selfplay.adar_selfplay_ray_trainer import RayAdaRSelfPlayTrainer
print('---IMPORT--- 所有Self-Play模块导入成功!')
"

if [ $? -ne 0 ]; then
    echo "---ERROR--- 模块导入失败!"
    exit 1
fi

# === 验证reward函数 ===
$CONDA_PYTHON -c "
import sys
sys.path.insert(0, '$VERL_DIR')
from recipe.adar_selfplay.reward_func import register_adar_reward, compute_score
register_adar_reward()
score = compute_score('The answer is \\\boxed{42}', '42.0')
print(f'---REWARD--- Test score: {score}')
assert score == 1.0, 'Reward function test failed!'
print('---REWARD--- Reward function verified OK')
"

if [ $? -ne 0 ]; then
    echo "---ERROR--- Reward function verification failed!"
    exit 1
fi

# === Stage4-Only测试训练 (标准GRPO) ===
echo "---TRAIN--- 开始Stage4-Only测试训练..."
cd "$VERL_DIR"

$CONDA_PYTHON -m recipe.adar_selfplay.run_adar_selfplay \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$TRAIN_DATA" \
    data.train_batch_size=16 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='AdaR-SelfPlay-test' \
    trainer.experiment_name='test_selfplay_t4only_0.5b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.default_local_dir="$CKPT_DIR" \
    trainer.test_freq=50 \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    adar_selfplay.enable_selfplay=False \
    2>&1 | tee "$LOG_DIR/test_selfplay_t4only_$(date +%Y%m%d_%H%M%S).log"

echo "---TEST--- Stage4-Only测试完成, exit code: $?"
