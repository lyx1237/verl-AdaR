# AdaR Self-Play

基于 [verl](https://github.com/volcengine/verl) 框架实现的 Self-Play 版 AdaR 训练系统。让一个小模型（~4B）同时承担数据增强和训练的角色，通过 4 阶段 pipeline 在训练过程中自动合成变体数学题并进行强化学习。

> 详细的设计文档和技术细节见 [doc.md](doc.md)。

## 背景

原版 AdaR 使用外部大模型（72B）离线合成变体问题数据，再用这些数据训练小模型。Self-Play 版将大模型替换为正在训练的小模型本身，实现了在线数据增强——模型一边生成题目变体，一边从自己生成的数据中学习。

## Pipeline

每个训练 step 执行以下 4 个阶段（所有阶段共享同一个 actor 模型）：

```
原始数学题 + CoT
      │
      ▼
  ┌─ Stage1: 提取题目模板 + 生成解题代码 ──→ 自动校验 ──→ 变量扰动
  │
  ├─ Stage2: 解答扰动后的题目 ──→ EVS 筛选（代码执行验证）
  │
  ├─ Stage3: 对通过筛选的题目做 Paraphrase 改写
  │
  └─ Stage4: 解答改写后的题目 ──→ 计算 Reward
      │
      ▼
  合并 4 阶段 loss，加权更新模型
```

各阶段的 reward 设计：
- **Stage1**: verify + 扰动 + EVS 全部通过 → 1，否则 → 0
- **Stage2**: 答案正确 → 1，否则 → 0（全错 group 不参与梯度更新）
- **Stage3**: `1 - 4*(acc-0.5)^2`，鼓励生成适当难度的改写
- **Stage4**: 答案正确 → 1，否则 → 0（全错 group 不参与梯度更新）

## 目录结构

```
verl/
├── recipe/adar_selfplay/       # Self-Play recipe（核心代码）
│   ├── run_adar_selfplay.py    #   训练入口
│   ├── adar_selfplay_ray_trainer.py  #   4 阶段 trainer
│   ├── adar_selfplay_reward.py #   reward 计算
│   ├── auto_pipeline.py        #   代码执行 / 校验 / 扰动 / EVS
│   ├── prompt_builder.py       #   各阶段 prompt 构造
│   ├── reward_func.py          #   reward 函数注册
│   ├── test_reward_logic.py    #   reward 单元测试
│   └── config/
│       └── adar_selfplay_trainer.yaml  #   Hydra 配置
├── scripts/adar/               # 训练脚本
│   ├── prepare_selfplay_data.py      #   数据预处理
│   └── *.sh                          #   各种训练/测试脚本
├── data/
│   ├── raw/                    # 原始数据（JSON）
│   └── selfplay/               # 预处理后数据（Parquet）
├── models/                     # 模型权重
├── ckpt/                       # 训练 checkpoint
├── logs/                       # 训练日志
├── verl/                       # verl 框架源码
└── doc.md                      # 设计文档
```

## 快速开始

### 1. 安装环境

```bash
conda create -n adar-verl python=3.10
conda activate adar-verl
pip install -e .
# 安装完成后确认：
python -c "import verl; print('verl OK')"
```

### 2. 准备数据

原始数据为 JSON，每条包含 `query`（数学题）、`chosen`（CoT 解答）、`answer`（数值答案）：

```json
{
  "query": "If 35 men working 5 hours a day earn Rs. 1715 per week, ...",
  "chosen": "To find out how much 11 men would earn, ...",
  "answer": 2695.0
}
```

转换为 verl 格式：

```bash
python scripts/adar/prepare_selfplay_data.py \
    data/raw/orca_80k.json \
    data/selfplay/train_selfplay_80k.parquet
```

### 3. 准备模型

将模型放到 `models/` 目录下（或创建软链接）：

```bash
ln -s /path/to/Qwen3-4B models/Qwen3-4B
```

### 4. 启动训练

```bash
# 编辑脚本中的 CUDA_VISIBLE_DEVICES 等环境配置，然后：
bash scripts/adar/adar_selfplay_qwen3_4b_full_80k_20260401.sh
```

或直接运行（最小示例）：

```bash
cd verl/
conda run -n lyx-verl python -m recipe.adar_selfplay.run_adar_selfplay \
    data.train_files=data/selfplay/train_selfplay_80k.parquet \
    data.val_files=data/selfplay/train_selfplay_80k.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=models/Qwen3-4B \
    trainer.n_gpus_per_node=4 \
    adar_selfplay.enable_selfplay=True
```

### 5. 监控

训练日志输出到 `logs/` 目录，同时上报 WandB（如配置）。关键指标：

| 指标 | 含义 |
|------|------|
| `selfplay/stage1_verify_pass_rate` | Stage1 模板+代码校验通过率 |
| `selfplay/stage2_avg_reward` | Stage2 解题正确率 |
| `selfplay/stage4_avg_accuracy` | Stage4 解答 paraphrase 题的正确率 |
| `selfplay/stage3_avg_reward` | Stage3 改写质量（越接近 1 表示难度越合适） |

## 配置参考

完整配置见 [`recipe/adar_selfplay/config/adar_selfplay_trainer.yaml`](recipe/adar_selfplay/config/adar_selfplay_trainer.yaml)，可通过命令行覆盖。

### 运行模式

| 配置 | 效果 |
|------|------|
| `adar_selfplay.enable_selfplay=False` | 标准 GRPO，仅 Stage4（baseline） |
| `adar_selfplay.enable_selfplay=True` | 完整 4 阶段 Self-Play |
| `adar_selfplay.enable_stage3_paraphrase=False` | 只做 Stage1 + Stage2 |
| `adar_selfplay.debug_inject_stage1=True` | 注入假 Stage1 结果，调试后续阶段 |

### 关键参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `n1` / `n2` / `n3` / `n4` / `n5` | 各阶段 rollout / 扰动次数 | 4 / 5 / 8 / 4 / 8 |
| `w1` / `w2` / `w3` / `w4` | 各阶段 loss 权重 | 0.2 / 0.3 / 0.2 / 0.3 |
| `max_template_code_length` | Stage1 prompt 最大长度 | 2048 |
| `max_solve_length` | Stage2/4 prompt 最大长度 | 2048 |
| `perturb_timeout` | 单样本扰动超时（秒） | 30 |

> **注意**: Self-Play 模式下 `actor_rollout_ref.rollout.n` 无效，rollout 次数由 `n1`~`n5` 独立控制，设为 1 即可。

## 运行测试

```bash
# reward 逻辑单元测试
python recipe/adar_selfplay/test_reward_logic.py
```
