### 原版AdaR

在原版AdaR框架下, 一个外部的大模型(72B)负责合成变体问题数据. 具体来说, 这个合成变体问题的过程分为以下步骤:
1. 生成题目模板和解题代码
2. 一个自动算法检查正确性, 包括变量是否一致, 程序是否可以执行, 执行结果是否与答案一致
3. 一个自动算法对题目施加扰动,产生变体问题
4. 大模型回答这些变体问题, 只要多个(n=5)结果中有和解题代码结果一致的, 就认为这个扰动是可以接受的
5. 大模型对通过的变体问题进行paraphrase
6. 收集数据, 用于训练内部的小模型

### Self-Play版

现在把外部的大模型(72B)替换为本地的小模型(~3B), 并且参与训练.
1. 给定一个问题, 模型提取模板和解题代码, rollout n_1 次, 此次的输入输出记为 T1[In_1,Out_1]
2. 自动校验
3. 自动扰动, rollout n_2次.
4. 对扰动后的变体问题, 让本地小模型尝试解答, rollout n_3 次, 此次的输入输出记为 T2[In_2, Out_2], 只要有和代码运行结果一致的, 就接受该扰动
5. 对通过的扰动做paraphrase, rollout n_4 次, 此次的输入输出记为 T3[In_3,Out_3]
6. 小模型尝试对paraphrase的问题进行回答, rollout n_5 次, 此次输入输出记为 T4[In_4,Out_4]
7. 计算reward, 记第6步某道题的正确率为acc, T4的reward= 1 if correct else 0, T3的reward=1-4*sqr(acc-0.5), 表示paraphrase适当的有点挑战性比较好. T2 的reward= 1 if correct else 0 , 但是如果不存在和代码运行一致的解答, 则舍弃这个批次, 不计算loss, 并且认为第一步的结果有误. T1的reward=1 if 步骤2,3,4均通过 else 0. 4个loss 都按照grpo的方式计算, token-mean-group-mean. 最后的总loss为4个loss带权相加, 权作为参数可设置.

## 概述

基于verl框架实现了Self-Play版AdaR训练pipeline。核心思路是在verl的RayPPOTrainer基础上，实现一个自定义的`RayAdaRSelfPlayTrainer`，在每个训练step中执行4阶段的self-play pipeline（T1→T2→T3→T4），共享同一个actor模型进行rollout和参数更新。

支持两种模式:
- **T4-only模式** (`enable_selfplay=False`): 退化为标准GRPO训练，已测试通过
- **完整Self-Play模式** (`enable_selfplay=True`): 执行4阶段pipeline，待测试

## 新增文件

### verl/recipe/adar_selfplay/ (verl框架侧, 6个新文件)

| 文件 | 说明 |
|------|------|
| `__init__.py` | 包初始化 |
| `auto_pipeline.py` | 核心pipeline逻辑，从scripts/adar重构而来 |
| `prompt_builder.py` | 各阶段prompt构造 + DataProto编码 |
| `adar_selfplay_reward.py` | 4阶段reward计算函数 |
| `adar_selfplay_ray_trainer.py` | 自定义trainer，继承RayPPOTrainer |
| `run_adar_selfplay.py` | Hydra入口脚本 |
| `config/adar_selfplay_trainer.yaml` | Hydra配置文件 |

### scripts/adar/ (训练脚本侧, 2个新文件)

| 文件 | 说明 |
|------|------|
| `prepare_selfplay_data.py` | 将原始JSON数据转换为verl parquet格式 |
| `test_adar_selfplay_0.5b_20260330.sh` | T4-only模式测试脚本 (已通过) |
| `test_adar_selfplay_full_0.5b_20260330.sh` | 0.5B完整测试 (已通过) |
| `test_adar_selfplay_qwen3_4b_20260330.sh` | Qwen3-4B完整测试(已通过) |

## 各文件详细说明

### 1. `auto_pipeline.py` — 核心pipeline逻辑

从`scripts/adar/`中的以下脚本重构提取:
- `auto_pipeline.py` → `SafeExecutor`, `parse_and_verify()`, `randomize_value()`, `randomize_code_once()`
- `check_evs.py` → `check_evs()`, `compute_evs_accuracy()`
- `extract_utils.py` → `extract_last_num()`, `extract_last_number_from_solution()`

主要类/函数:
- **`SafeExecutor`**: 基于subprocess的安全代码执行器，支持超时控制
- **`parse_and_verify()`**: 从T1生成结果中提取template和code，验证正确性
- **`perturb_variables()`**: 对验证通过的代码进行N次变量扰动，生成新问题
- **`check_evs()`**: EVS检验 — 执行扰动后代码确认输出一致
- **`compute_evs_accuracy()`**: 用模型解答计算EVS准确率

### 2. `prompt_builder.py` — Prompt构造器

**`PromptBuilder`类**:
- `build_t1_prompts()`: 原始题目 + CoT → "提取模板和代码" 指令
- `build_t2_prompts()`: 扰动题 + 代码提示 → "解答" 指令
- `build_t3_prompts()`: 题目 → "改写这道题" 指令
- `build_t4_prompts()`: paraphrase题 → "解答" 指令
- `_encode_prompts()`: 文本→DataProto (input_ids, attention_mask, position_ids, uid, raw_prompt_ids)

编码细节: 左截断、左padding、position_ids从0开始计数有效token。

### 3. `adar_selfplay_reward.py` — 多阶段Reward

| 阶段 | 函数 | Reward公式 |
|------|------|------------|
| T1 | `compute_t1_reward()` | 二元: verify+perturb+evs全部通过=1, 否则=0 |
| T2 | `compute_t2_reward()` | 二元: 答案正确=1, 否则=0 |
| T3 | `compute_t3_reward()` | `1 - 4*(acc-0.5)^2`, acc为T4解答准确率, 鼓励中等难度 |
| T4 | `compute_t4_reward()` | 二元: 答案正确=1, 否则=0 |

### 4. `adar_selfplay_ray_trainer.py` — 自定义Trainer

**`RayAdaRSelfPlayTrainer(RayPPOTrainer)`**:

核心方法:
- `fit()`: 主训练循环，根据`enable_selfplay`配置选择模式
- `_run_selfplay_pipeline()`: 完整4阶段pipeline
  1. T1: generate → parse_and_verify → perturb → evs筛选
  2. T2: generate解答扰动题 → 计算reward
  3. T3: generate paraphrase → T4解答 → 计算T3 reward
  4. T4: 计算T4 reward
  5. 合并4阶段batch，加权advantages，更新actor
- `_run_standard_grpo()`: T4-only标准GRPO
- `_generate_for_stage()`: 通用rollout接口
- `_compute_advantage_for_stage()`: GRPO advantage = (reward - mean) / std
- `_merge_and_update()`: 合并多阶段DataProto，按权重缩放advantage

可配置开关:
- `enable_selfplay`: 是否启用self-play (False退化为标准GRPO)
- `enable_t2_evs`: 是否启用T2+EVS筛选阶段
- `enable_t3_paraphrase`: 是否启用T3+paraphrase阶段

### 5. `run_adar_selfplay.py` — 入口脚本

- `@hydra.main`入口，加载`adar_selfplay_trainer.yaml`配置
- Ray初始化: 清除`RAY_ADDRESS`环境变量，清理stale `/tmp/ray` session
- `TaskRunner` (Ray remote): 初始化tokenizer, workers, reward manager, trainer

### 6. `config/adar_selfplay_trainer.yaml` — 配置

继承verl的`ppo_trainer`默认配置，添加`adar_selfplay`专有配置:
- rollout次数: n1=4, n2=5, n3=8, n4=4, n5=8
- loss权重: w1=0.2, w2=0.3, w3=0.2, w4=0.3
- 扰动参数: alpha=5, timeout=30s, code_timeout=2s
- 序列长度: template_code=2048, solve=2048, paraphrase=1024

### 7. `prepare_selfplay_data.py` — 数据准备

将`orca_200.json`转换为verl parquet格式:
- `prompt`: system + user messages
- `reward_model`: `{style: "rule", ground_truth: answer}`
- `extra_info`: `{id, query, chosen, answer}` — self-play pipeline需要的原始数据

## 未修改的文件

本次实现完全基于增量新增文件，未修改verl框架或AdaR项目中的任何现有文件。

