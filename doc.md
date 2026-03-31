# AdaR Self-Play 实现文档

日期: 2026-03-30
对应计划: `TODO0330.md`

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
| `auto_pipeline.py` | 核心pipeline逻辑，从AdaR/scripts重构而来 |
| `prompt_builder.py` | 各阶段prompt构造 + DataProto编码 |
| `adar_selfplay_reward.py` | 4阶段reward计算函数 |
| `adar_selfplay_ray_trainer.py` | 自定义trainer，继承RayPPOTrainer |
| `run_adar_selfplay.py` | Hydra入口脚本 |
| `config/adar_selfplay_trainer.yaml` | Hydra配置文件 |

### AdaR/scripts/ (AdaR项目侧, 2个新文件)

| 文件 | 说明 |
|------|------|
| `prepare_selfplay_data.py` | 将原始JSON数据转换为verl parquet格式 |
| `test_adar_selfplay_0.5b_20260330.sh` | T4-only模式测试脚本 (已通过) |

## 各文件详细说明

### 1. `auto_pipeline.py` — 核心pipeline逻辑

从`AdaR/scripts/`中的以下脚本重构提取:
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

## 测试结果

### T4-only模式 (标准GRPO)
- 服务器: A6000-4, GPU 4,5
- 模型: Qwen2.5-0.5B-Instruct
- 配置: batch_size=16, max_prompt/response_length=128, n=4
- 结果: 成功完成11个training steps，checkpoint保存正常
- 脚本: `test_adar_selfplay_0.5b_20260330.sh` (已标记TESTED: YES)

### 完整Self-Play模式
- 服务器: A6000-4, GPU 0,1
- 模型: Qwen2.5-0.5B-Instruct
- 配置: batch_size=8, max_prompt/response_length=256, n1=n2=n3=n4=n5=2
- 结果: 成功完成24个training steps，checkpoint保存正常
- T1验证通过率: 0% (0.5B模型太小，无法生成有效模板+代码，预期行为)
- 由于T1全部失败，T2/T3/T4被跳过，仅T1 batch参与更新
- 脚本: `test_adar_selfplay_full_0.5b_20260330.sh` (已标记TESTED: YES)

### 完整Self-Play模式 (Qwen3-4B)
- 服务器: A6000-6, GPU 3,4
- 模型: Qwen3-4B
- 配置: batch_size=8, max_prompt/response_length=512, n1=n2=n3=n4=n5=2, param_offload=True, optimizer_offload=True, gpu_memory_utilization=0.5
- 结果: 成功完成25个training steps，checkpoint保存正常
- T1验证通过率: ~44% (11/25 steps触发完整T1→T2→T3→T4 pipeline)
- 4B模型能够生成有效的模板+代码，成功触发完整pipeline
- 脚本: `test_adar_selfplay_qwen3_4b_20260330.sh` (已标记TESTED: YES)

### 测试过程中修复的问题
1. **flashinfer JIT编译失败**: 其他用户的Ray集群导致worker使用错误CUDA路径。通过`_temp_dir`隔离Ray session解决。
2. **DataProto union冲突**: `generate_sequences`返回的`input_ids`与输入不同。改为直接使用gen_output，仅合并non_tensor数据。
3. **ref_log_prob缺失**: Self-Play路径中`_compute_advantage_for_stage`未计算reference policy log_prob。添加了ref_log_prob计算。
4. **多阶段batch序列长度不匹配**: 不同阶段(T1/T2/T3/T4)生成的序列长度不同，`DataProto.concat`要求所有tensor的dim-1一致。修复: 按key计算最大dim-1，对较短tensor进行zero-padding。
5. **DataProto比较错误**: `valid_batches.index(b)`触发TensorDict的`__eq__`导致shape mismatch。修复: 使用`enumerate`替代。
6. **non_tensor_batch维度不匹配**: 不同阶段的non_tensor_batch结构不同(2D vs 1D数组)。修复: concat前清除non_tensor_batch。
7. **CUDA OOM (Qwen3-4B)**: 4B模型在2x48GB GPU上显存不足。修复: 启用param_offload和optimizer_offload，降低gpu_memory_utilization到0.5。
8. **常量变量导致扰动拒绝**: `auto_pipeline.py`的`randomize_code_once()`中，代码里无模板占位符的变量(如`days_in_week=7`)会导致整个样本被拒绝。修复: 与`AdaR/scripts/perturb_variables.py`同步，将无占位符变量视为常量跳过，只扰动有占位符的变量。
