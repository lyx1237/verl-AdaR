# AdaR Self-Play Reward 逻辑验证报告

日期: 2026-04-01

## 1. 背景

根据 `TODO.md` 的描述，对 `adar_selfplay_ray_trainer.py` 和 `adar_selfplay_reward.py` 进行了重大重构，修正了 pipeline 的核心逻辑。主要改动：

1. **Stage1 per-rollout 评估**: 不再只取第一个通过的 rollout（旧代码有 `break`），而是对每个 rollout 独立评估 reward
2. **Stage1 reward 依赖 EVS**: parse_and_verify 未通过→0，所有扰动 EVS 失败→0，至少一个扰动通过→1
3. **Stage 1.5 per-rollout 扰动**: 对每个通过 verify 的 rollout 分别做扰动，而非只对每个问题做一次
4. **Stage2/Stage4 training_mask**: 全错 group 的所有条目不参与参数更新（response_mask 和 advantage 置 0）
5. **Stage3 收集所有通过的扰动**: 不再只取每个 prompt 的第一个，而是所有通过 EVS 的扰动都进入 paraphrase
6. **Stage3 reward 通过 t3_valid_map 映射**: 正确处理空 paraphrase 跳过的情况

## 2. 验证方法

采用三层验证：

### 2.1 单元测试（手工构造数据）

文件: `recipe/adar_selfplay/test_reward_logic.py`

用手工构造的确定性数据验证每个 reward 函数的计算正确性，共 7 组测试 32 个断言：

| 测试组 | 验证内容 | 断言数 |
|--------|---------|--------|
| Stage1 Reward | 8题×2rollout=16个样本的 per-rollout reward | 2 |
| Stage2 Reward + Mask | 12扰动×3次=36个response，reward + training_mask | 5 |
| Stage4 Reward + Mask | 5问题×4次=20个response，accuracy + mask | 4 |
| Stage3 Reward | `1-4*(acc-0.5)^2` 公式在不同 acc 下的值 | 1 |
| Stage3 valid_map 映射 | 空 paraphrase 跳过 + Stage4 accuracy 回映射 | 3 |
| Training Mask 应用 | response_mask × mask, advantage × mask | 8 |
| 端到端 Pipeline | 从 Stage1→Stage1.5→Stage2→Stage3→Stage4 的完整数据流 | 9 |

运行方式:
```bash
conda run -n lyx-verl python /home/zfs01/liyx/verl/recipe/adar_selfplay/test_reward_logic.py
```

### 2.2 集成测试（0.5B 模型 + debug 注入）

在 5880-1 GPU4 上，使用 Qwen2.5-0.5B-Instruct + `debug_inject_t1=True`（注入假 Stage1 通过结果），让真实模型 rollout Stage2/Stage3/Stage4。

脚本: `scripts/adar/test_reward_verify_0.5b_20260331.sh`

这样可以：
- 确保完整 pipeline Stage1→Stage1.5→Stage2→Stage3→Stage4 端到端运行无报错
- 用真实模型输出验证 reward 计算（提取数值 → 比对 expected → 给分）
- 验证 training_mask 在 advantage 计算中的应用
- 验证 loss 和 grad_norm 的合理性

### 2.3 集成测试（4B 模型 + debug 注入）

在 3090-5 4×GPU 上，使用 Qwen3-4B + `debug_inject_t1=True`，4B 模型的解题能力更强，可以产生更有意义的 Stage2 正确/错误分布。

脚本: `scripts/adar/test_reward_verify_4b_3090_20260401.sh`

## 3. 单元测试结果

```
============================================================
测试总结
============================================================
  ✓ PASS Stage1 Reward
  ✓ PASS Stage2 Reward + Mask
  ✓ PASS Stage4 Reward + Mask
  ✓ PASS Stage3 Reward (compute_stage3_reward)
  ✓ PASS Stage3 valid_map 映射
  ✓ PASS Training Mask 应用
  ✓ PASS 端到端Pipeline

  全部测试通过!
```

32/32 断言通过。

## 4. 0.5B 集成测试结果

### 4.1 无 debug 注入（真实 Stage1）

服务器: 5880-1, GPU4, Qwen2.5-0.5B-Instruct

结果: 2 步完成，Stage1 verify = 0/16（0.5B 模型太弱无法生成 template+code），只有 Stage1 阶段运行。

| 指标 | Step 1 | Step 2 |
|------|--------|--------|
| Stage1 verify pass rate | 0/16 | 0/16 |
| Stage1 avg reward | 0.0 | 0.0 |
| pg_loss | 0.0 | 0.0 |
| kl_loss | 0.00012 | 0.00024 |

结论: pipeline 框架正确运行，但模型能力不足导致无法触发后续阶段。

### 4.2 debug 注入模式

服务器: 5880-1, GPU4, Qwen2.5-0.5B-Instruct, `debug_inject_t1=True`

注入策略: rollout0 对所有问题通过，rollout1 只对偶数问题通过。

结果: 3 步完整 Stage1→Stage2→Stage3→Stage4 pipeline 运行成功。

```
step:3 指标:
- selfplay/t1_verify_pass_rate: 0.75
- selfplay/evs_pass_rate: 0.87
- selfplay/t2_avg_reward: 0.65
- selfplay/t2_masked_out_groups: 3
- selfplay/t4_avg_reward: 0.30
- selfplay/t4_masked_out_groups: 26
- selfplay/t3_avg_reward: 0.10
- actor/pg_loss: -0.0016
- actor/grad_norm: 1.11
```

手动验算样例:

**Stage1:**
- 扰动0: expected=357, 尝试0提取357 → reward=1 ✓
- 扰动7: expected=264, 两次提取22和12 → 全错 → mask=0 ✓

**Stage4:**
- T4问题0: expected=357, 提取1800和10 → 全错 → acc=0, mask=0 ✓
- T4问题2: expected=630, 提取630和630 → 全对 → acc=1.0, mask=1 ✓

## 5. Qwen3-4B 集成测试结果 (详细)

服务器: 3090-5, 4×RTX 3090 24GB, Qwen3-4B, `debug_inject_t1=True`, `use_kl_loss=False`

日志文件: `AdaR/logs/test_reward_4b_3090_4gpu_v3.log`

### 5.0 测试配置

- 8 道简单数学题，n1=2, n2=2, n3=2, n4=2, n5=2
- debug_inject_t1: rollout0 对所有 8 个问题通过，rollout1 只对偶数问题 (0,2,4,6) 通过
- 因此 Stage1 batch=16 (8×2)，其中 12 个 rollout 注入为"通过"
- Stage2/Stage3/Stage4 由 Qwen3-4B 真实 rollout 生成

### 5.1 Step 1 总览指标

```
selfplay/t1_verify_pass_rate:  0.75        (12/16 rollout 通过)
selfplay/t1_verify_prompt_rate: 1.0        (8/8 个问题至少有一个 rollout 通过)
selfplay/perturb_pass_rate:    1.0         (12/12 个通过 rollout 都成功生成了扰动)
selfplay/evs_pass_rate:        0.75        (18/24 个扰动通过 EVS)
selfplay/evs_rollout_rate:     0.917       (11/12 个 rollout 至少有一个扰动通过)
selfplay/paraphrase_count:     36          (18 个扰动 × n4=2 = 36 个 paraphrase)
selfplay/t1_avg_reward:        0.6875      (11/16 = 0.6875)
selfplay/t2_avg_reward:        0.50        (24/48 次尝试正确)
selfplay/t2_masked_out_groups: 6           (6 个全错扰动被排除)
selfplay/t4_avg_reward:        0.014       (1/72 次回答正确)
selfplay/t4_avg_accuracy:      0.014
selfplay/t4_masked_out_groups: 35          (35/36 个 group 全错)
selfplay/t3_avg_reward:        0.028       (只有 1 个 Stage4 group 有 acc>0)
actor/pg_loss:                 0.00023
actor/pg_clipfrac:             0.007
actor/ppo_kl:                  0.008
actor/grad_norm:               0.144
```

### 5.2 Stage1 Reward 逐条验算

**Reward 规则**: verify 未通过→0，verify 通过但所有扰动 EVS 失败→0，至少一个扰动通过 EVS→1

debug_inject 注入策略: rollout0 全部通过 verify，rollout1 仅偶数问题 (0,2,4,6) 通过。

日志原文:
```
问题0: A baker makes 15 cakes per day...  (ans=60.0)
  rollout0: verify=✓, has_pert=True, evs_pass=1, reward=1
  rollout1: verify=✓, has_pert=True, evs_pass=2, reward=1
问题1: A store sells apples for 3 dollars each...  (ans=15.0)
  rollout0: verify=✓, has_pert=True, evs_pass=1, reward=1
  rollout1: verify=✗, has_pert=False, evs_pass=0, reward=0
问题2: If a shirt costs 25 dollars...  (ans=20.0)
  rollout0: verify=✓, has_pert=True, evs_pass=1, reward=1
  rollout1: verify=✓, has_pert=True, evs_pass=2, reward=1
问题3: A car travels at a speed of 60 km/h...  (ans=180.0)
  rollout0: verify=✓, has_pert=True, evs_pass=2, reward=1
  rollout1: verify=✗, has_pert=False, evs_pass=0, reward=0
问题4: A triangle has a base of 10 meters...  (ans=30.0)
  rollout0: verify=✓, has_pert=True, evs_pass=2, reward=1
  rollout1: verify=✓, has_pert=True, evs_pass=0, reward=0
问题5: Sarah has 20 candies...  (ans=13.0)
  rollout0: verify=✓, has_pert=True, evs_pass=2, reward=1
  rollout1: verify=✗, has_pert=False, evs_pass=0, reward=0
问题6: A rectangle has a length of 8 cm...  (ans=48.0)
  rollout0: verify=✓, has_pert=True, evs_pass=2, reward=1
  rollout1: verify=✓, has_pert=True, evs_pass=2, reward=1
问题7: John has 4 boxes...  (ans=48.0)
  rollout0: verify=✓, has_pert=True, evs_pass=1, reward=1
  rollout1: verify=✗, has_pert=False, evs_pass=0, reward=0
```

逐条验算:

| 问题 | rollout | verify | has_pert | evs_pass | 期望reward | 实际reward | 推理过程 |
|------|---------|--------|----------|----------|-----------|-----------|---------|
| 0 (baker, 偶数) | 0 | ✓ | True | 1 | **1** | 1 ✓ | verify通过 + 至少1个EVS通过 → 1 |
| 0 | 1 | ✓ | True | 2 | **1** | 1 ✓ | 偶数问题rollout1通过 + EVS通过 → 1 |
| 1 (apples, 奇数) | 0 | ✓ | True | 1 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 1 | 1 | ✗ | False | 0 | **0** | 0 ✓ | 奇数问题rollout1 verify失败 → 0 |
| 2 (shirt, 偶数) | 0 | ✓ | True | 1 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 2 | 1 | ✓ | True | 2 | **1** | 1 ✓ | 偶数 + EVS通过 → 1 |
| 3 (car, 奇数) | 0 | ✓ | True | 2 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 3 | 1 | ✗ | False | 0 | **0** | 0 ✓ | 奇数rollout1 verify失败 → 0 |
| **4 (triangle, 偶数)** | 0 | ✓ | True | 2 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| **4** | **1** | **✓** | **True** | **0** | **0** | **0 ✓** | **verify通过但EVS全部失败 → 0 (关键case)** |
| 5 (candies, 奇数) | 0 | ✓ | True | 2 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 5 | 1 | ✗ | False | 0 | **0** | 0 ✓ | 奇数rollout1 verify失败 → 0 |
| 6 (rectangle, 偶数) | 0 | ✓ | True | 2 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 6 | 1 | ✓ | True | 2 | **1** | 1 ✓ | 偶数 + EVS通过 → 1 |
| 7 (boxes, 奇数) | 0 | ✓ | True | 1 | **1** | 1 ✓ | verify通过 + EVS通过 → 1 |
| 7 | 1 | ✗ | False | 0 | **0** | 0 ✓ | 奇数rollout1 verify失败 → 0 |

**16/16 全部正确**。

特别注意 **问题4 rollout1**: 这是唯一一个 verify 通过、扰动也生成了、但 EVS 全部未通过的 case。按 TODO.md 的逻辑 "如果 n_2 次扰动全部失败, 那么我们认为没有扰动可以成功了, 换言之就是模板提取的就是错的, 所以我们回过头给提取模板的那个步骤 0 reward"——实际输出 reward=0，**逻辑完全符合设计**。

统计: reward=1 有 11 个，reward=0 有 5 个，平均 = 11/16 = 0.6875，与日志 `t1_avg_reward: 0.6875` 一致 ✓

### 5.3 Stage2 Reward + Training Mask 逐条验算

**Reward 规则**: 模型提取的数值匹配 expected → reward=1, 否则 → 0。全错 group → training_mask=0。

Stage 1.5 对 12 个通过 verify 的 rollout 分别做了 n2=2 次扰动，全部成功，共 24 个扰动。EVS 检验中 18 个通过，6 个失败（全错的扰动 mask=0）。

日志原文 (完整 24 个扰动):
```
扰动0:  expected=408.0,  EVS=✓, mask=1  | 尝试0: 408.0→reward=1 | 尝试1: 3.0→reward=0
扰动1:  expected=600.0,  EVS=✗, mask=0  | 尝试0: 8.0→reward=0   | 尝试1: 3.0→reward=0
扰动2:  expected=612.0,  EVS=✓, mask=1  | 尝试0: 612.0→reward=1 | 尝试1: 612.0→reward=1
扰动3:  expected=299.0,  EVS=✓, mask=1  | 尝试0: 299.0→reward=1 | 尝试1: 23.0→reward=0
扰动4:  expected=66.0,   EVS=✓, mask=1  | 尝试0: 66.0→reward=1  | 尝试1: 66.0→reward=1
扰动5:  expected=58.0,   EVS=✗, mask=0  | 尝试0: 5.0→reward=0   | 尝试1: 2.0→reward=0
扰动6:  expected=0.0,    EVS=✗, mask=0  | 尝试0: 11.0→reward=0  | 尝试1: 100.0→reward=0
扰动7:  expected=32.0,   EVS=✓, mask=1  | 尝试0: 100.0→reward=0 | 尝试1: 32.0→reward=1
扰动8:  expected=12.0,   EVS=✓, mask=1  | 尝试0: 12.0→reward=1  | 尝试1: 20.0→reward=0
扰动9:  expected=48.0,   EVS=✓, mask=1  | 尝试0: 48.0→reward=1  | 尝试1: 20.0→reward=0
扰动10: expected=1216.0, EVS=✓, mask=1  | 尝试0: 1216.0→reward=1| 尝试1: 1216.0→reward=1
扰动11: expected=3212.0, EVS=✓, mask=1  | 尝试0: 292.0→reward=0 | 尝试1: 3212.0→reward=1
扰动12: expected=735.0,  EVS=✓, mask=1  | 尝试0: 735.0→reward=1 | 尝试1: 735.0→reward=1
扰动13: expected=208.0,  EVS=✓, mask=1  | 尝试0: 208.0→reward=1 | 尝试1: 26.0→reward=0
扰动14: expected=444.0,  EVS=✗, mask=0  | 尝试0: 37.0→reward=0  | 尝试1: 888.0→reward=0
扰动15: expected=154.0,  EVS=✗, mask=0  | 尝试0: 2.0→reward=0   | 尝试1: 14.0→reward=0
扰动16: expected=39.0,   EVS=✓, mask=1  | 尝试0: 39.0→reward=1  | 尝试1: 39.0→reward=1
扰动17: expected=47.0,   EVS=✓, mask=1  | 尝试0: 47.0→reward=1  | 尝试1: 47.0→reward=1
扰动18: expected=841.0,  EVS=✓, mask=1  | 尝试0: 29.0→reward=0  | 尝试1: 841.0→reward=1
扰动19: expected=957.0,  EVS=✓, mask=1  | 尝试0: 29.0→reward=0  | 尝试1: 957.0→reward=1
扰动20: expected=98.0,   EVS=✓, mask=1  | 尝试0: 98.0→reward=1  | 尝试1: 14.0→reward=0
扰动21: expected=6.0,    EVS=✓, mask=1  | 尝试0: 2.0→reward=0   | 尝试1: 6.0→reward=1
扰动22: expected=396.0,  EVS=✓, mask=1  | 尝试0: 396.0→reward=1 | 尝试1: 66.0→reward=0
扰动23: expected=180.0,  EVS=✗, mask=0  | 尝试0: 1.0→reward=0   | 尝试1: 10.0→reward=0
```

逐条验算（抽样关键 case）:

| 扰动 | expected | 提取值 | 匹配? | 期望reward | 实际reward | 验证 |
|------|----------|--------|-------|-----------|-----------|------|
| 扰动0 尝试0 | 408.0 | 408.0 | `|408-408|<0.001` ✓ | 1 | 1 | ✓ |
| 扰动0 尝试1 | 408.0 | 3.0 | `|3-408|=405` ✗ | 0 | 0 | ✓ |
| 扰动1 尝试0 | 600.0 | 8.0 | ✗ | 0 | 0 | ✓ |
| 扰动1 尝试1 | 600.0 | 3.0 | ✗ | 0 | 0 | ✓ |
| 扰动6 尝试0 | 0.0 | 11.0 | `|11-0|=11` ✗ | 0 | 0 | ✓ |
| 扰动7 尝试1 | 32.0 | 32.0 | ✓ | 1 | 1 | ✓ |
| 扰动11 尝试0 | 3212.0 | 292.0 | ✗ | 0 | 0 | ✓ |
| 扰动11 尝试1 | 3212.0 | 3212.0 | ✓ | 1 | 1 | ✓ |
| 扰动14 尝试1 | 444.0 | 888.0 | `|888-444|=444` ✗ | 0 | 0 | ✓ (注: 888是444的两倍, 模型做了2倍错误) |

**Training mask 验算**:

全错 group（n3=2次尝试均未匹配 expected）:
- 扰动1: [0,0] → **mask=0** ✓ (expected=600, 提取 8 和 3)
- 扰动5: [0,0] → **mask=0** ✓ (expected=58, 提取 5 和 2)
- 扰动6: [0,0] → **mask=0** ✓ (expected=0, 提取 11 和 100)
- 扰动14: [0,0] → **mask=0** ✓ (expected=444, 提取 37 和 888)
- 扰动15: [0,0] → **mask=0** ✓ (expected=154, 提取 2 和 14)
- 扰动23: [0,0] → **mask=0** ✓ (expected=180, 提取 1 和 10)

共 6 个全错 group 被 mask，与日志 `t2_masked_out_groups: 6` 一致 ✓

日志中 `---ADV--- Stage2: training_mask排除了12/48个样本`：6 groups × 2 尝试/group = 12，正确 ✓

总 reward=1 的数量: 手动统计 = 24，总尝试=48，avg=24/48=0.50，与日志 `t2_avg_reward: 0.50` 一致 ✓

### 5.4 Stage4 Reward + Training Mask 逐条验算

**Reward 规则**: 提取数值匹配 expected → reward=1, 否则 → 0。group 内 n5=2 次全错 → mask=0。

Stage3 将 18 个通过 EVS 的扰动各做 n4=2 次 paraphrase，产生 36 个变体问题。Stage4 对每个问题 n5=2 次回答，共 72 个 response。

日志原文 (全部 36 个 Stage4 group):
```
T4问题0  (t3 entry (0,0)):  expected=408.0,  acc=0.00, mask=0
  回答0: extracted=17.0→0   回答1: extracted=24.0→0
T4问题1  (t3 entry (0,1)):  expected=408.0,  acc=0.00, mask=0
  回答0: extracted=17.0→0   回答1: extracted=17.0→0
T4问题2  (t3 entry (1,0)):  expected=612.0,  acc=0.00, mask=0
  回答0: extracted=4.0→0    回答1: extracted=18.0→0
T4问题3  (t3 entry (1,1)):  expected=612.0,  acc=0.00, mask=0
  回答0: extracted=34.0→0   回答1: extracted=18.0→0
T4问题4  (t3 entry (2,0)):  expected=299.0,  acc=0.00, mask=0
  回答0: extracted=23.0→0   回答1: extracted=13.0→0
T4问题5  (t3 entry (2,1)):  expected=299.0,  acc=0.00, mask=0
  回答0: extracted=13.0→0   回答1: extracted=13.0→0
T4问题6  (t3 entry (3,0)):  expected=66.0,   acc=0.00, mask=0
  回答0: extracted=3.0→0    回答1: extracted=15.0→0
T4问题7  (t3 entry (3,1)):  expected=66.0,   acc=0.00, mask=0
  回答0: extracted=6.0→0    回答1: extracted=11.0→0
T4问题8  (t3 entry (4,0)):  expected=32.0,   acc=0.00, mask=0
  回答0: extracted=40.0→0   回答1: extracted=20.0→0
T4问题9  (t3 entry (4,1)):  expected=32.0,   acc=0.00, mask=0
  回答0: extracted=20.0→0   回答1: extracted=40.0→0
T4问题10 (t3 entry (5,0)):  expected=12.0,   acc=0.00, mask=0
  回答0: extracted=20.0→0   回答1: extracted=15.0→0
T4问题11 (t3 entry (5,1)):  expected=12.0,   acc=0.00, mask=0
  回答0: extracted=15.0→0   回答1: extracted=20.0→0
T4问题12 (t3 entry (6,0)):  expected=48.0,   acc=0.00, mask=0
  回答0: extracted=20.0→0   回答1: extracted=20.0→0
T4问题13 (t3 entry (6,1)):  expected=48.0,   acc=0.00, mask=0
  回答0: extracted=20.0→0   回答1: extracted=20.0→0
T4问题14 (t3 entry (7,0)):  expected=1216.0, acc=0.00, mask=0
  回答0: extracted=8.0→0    回答1: extracted=8.0→0
T4问题15 (t3 entry (7,1)):  expected=1216.0, acc=0.00, mask=0
  回答0: extracted=8.0→0    回答1: extracted=8.0→0
T4问题16 (t3 entry (8,0)):  expected=3212.0, acc=0.00, mask=0
  回答0: extracted=11.0→0   回答1: extracted=292.0→0
T4问题17 (t3 entry (8,1)):  expected=3212.0, acc=0.00, mask=0
  回答0: extracted=11.0→0   回答1: extracted=11.0→0
T4问题18 (t3 entry (9,0)):  expected=735.0,  acc=0.00, mask=0
  回答0: extracted=215.0→0  回答1: extracted=12.0→0
T4问题19 (t3 entry (9,1)):  expected=735.0,  acc=0.00, mask=0
  回答0: extracted=35.0→0   回答1: extracted=35.0→0
T4问题20 (t3 entry (10,0)): expected=208.0,  acc=0.00, mask=0
  回答0: extracted=26.0→0   回答1: extracted=26.0→0
T4问题21 (t3 entry (10,1)): expected=208.0,  acc=0.00, mask=0
  回答0: extracted=16.0→0   回答1: extracted=26.0→0
T4问题22 (t3 entry (11,0)): expected=39.0,   acc=0.00, mask=0
  回答0: extracted=5.0→0    回答1: extracted=19.0→0
T4问题23 (t3 entry (11,1)): expected=39.0,   acc=0.00, mask=0
  回答0: extracted=1.0→0    回答1: extracted=19.0→0
T4问题24 (t3 entry (12,0)): expected=47.0,   acc=0.00, mask=0
  回答0: extracted=2.0→0    回答1: extracted=28.0→0
T4问题25 (t3 entry (12,1)): expected=47.0,   acc=0.00, mask=0
  回答0: extracted=28.0→0   回答1: extracted=28.0→0
T4问题26 (t3 entry (13,0)): expected=841.0,  acc=0.00, mask=0
  回答0: extracted=29.0→0   回答1: extracted=29.0→0
T4问题27 (t3 entry (13,1)): expected=841.0,  acc=0.00, mask=0
  回答0: extracted=2.0→0    回答1: extracted=29.0→0
T4问题28 (t3 entry (14,0)): expected=957.0,  acc=0.00, mask=0
  回答0: extracted=33.0→0   回答1: extracted=33.0→0
T4问题29 (t3 entry (14,1)): expected=957.0,  acc=0.00, mask=0
  回答0: extracted=33.0→0   回答1: extracted=29.0→0
T4问题30 (t3 entry (15,0)): expected=98.0,   acc=0.00, mask=0
  回答0: extracted=7.0→0    回答1: extracted=7.0→0
T4问题31 (t3 entry (15,1)): expected=98.0,   acc=0.00, mask=0
  回答0: extracted=14.0→0   回答1: extracted=14.0→0
T4问题32 (t3 entry (16,0)): expected=6.0,    acc=0.00, mask=0
  回答0: extracted=2.0→0    回答1: extracted=2.0→0
★ T4问题33 (t3 entry (16,1)): expected=6.0,  acc=0.50, mask=1
  回答0: extracted=6.0→1    回答1: extracted=2.0→0
T4问题34 (t3 entry (17,0)): expected=396.0,  acc=0.00, mask=0
  回答0: extracted=66.0→0   回答1: extracted=66.0→0
T4问题35 (t3 entry (17,1)): expected=396.0,  acc=0.00, mask=0
  回答0: extracted=66.0→0   回答1: extracted=66.0→0
```

**关键验算**:

1. **T4问题33 (唯一正确的)**: expected=6.0, 回答0 提取 6.0 → `|6-6|<0.001` → reward=1, 回答1 提取 2.0 → reward=0。acc=1/2=0.50, mask=1 ✓
2. **T4问题8**: expected=32.0, 回答 40 和 20 → 全错 → acc=0.00, mask=0 ✓ (注: 40 和 20 分别是原价和折后价, 模型没有理解扰动后的折扣率)
3. **T4问题14**: expected=1216.0, 两个回答都提取 8 → 全错 → mask=0 ✓ (模型在回答 paraphrase 时输出了原始参数而非计算结果)
4. **T4问题34**: expected=396.0, 两个回答都提取 66 → 全错 → mask=0 ✓ (66=6×11 是单箱数量, 模型没乘以箱数)

**统计验算**:
- reward=1 的数量: 仅 T4问题33 的回答0 → 总共 1 个
- 总回答数: 36×2=72
- avg_reward = 1/72 = 0.01389, 与日志 `t4_avg_reward: 0.014` 一致 ✓
- 全错 group 数: 35 (除了问题33), 与日志 `t4_masked_out_groups: 35` 一致 ✓
- masked 样本数: 35×2=70, 与日志 `---ADV--- Stage4: training_mask排除了70/72个样本` 一致 ✓

### 5.5 Stage3 Reward 逐条验算

**Reward 规则**: `reward = max(0, 1 - 4*(acc-0.5)^2)`，其中 acc 是对应 Stage4 group 的正确率。空 paraphrase（未产生 Stage4 问题的）reward=0。

Stage3 batch 共 36 个 entry (18 个源扰动 × n4=2 个 paraphrase rollout)。

只有 Stage4 问题33 有非零 accuracy (acc=0.50)，它对应 t3 entry (16,1)，即 t3_batch[33]。

日志原文:
```
t3_batch[0]:  reward=0.0000
t3_batch[1]:  reward=0.0000
...
t3_batch[32]: reward=0.0000
t3_batch[33]: reward=1.0000    ★
t3_batch[34]: reward=0.0000
t3_batch[35]: reward=0.0000
```

**验算 t3_batch[33]**:
- 对应 Stage4 问题33, acc=0.50
- reward = 1 - 4*(0.50-0.50)^2 = 1 - 0 = **1.0** ✓

**验算其他 entry (以 t3_batch[0] 为例)**:
- 对应 t3 entry (0,0) → Stage4 问题0, acc=0.00
- reward = 1 - 4*(0.00-0.50)^2 = 1 - 4*0.25 = 1 - 1 = **0.0** ✓

**验算 t3_batch[32]**:
- 对应 t3 entry (16,0) → Stage4 问题32, acc=0.00
- reward = 1 - 4*(0.00-0.50)^2 = 0.0 ✓ (注: 同一源扰动的两个 rollout, rollout0 对应的 Stage4 全错, rollout1 对应的 Stage4 有一个正确)

统计: avg_reward = 1/36 = 0.0278, 与日志 `t3_avg_reward: 0.028` 一致 ✓

### 5.6 Training Mask 在 Advantage 中的应用

日志:
```
---ADV--- Stage2: training_mask排除了12/48个样本
---ADV--- Stage4: training_mask排除了70/72个样本
```

验算:
- Stage2: 6 个全错 group × n3=2 = **12** 个样本被排除，总 48 个样本 ✓
- Stage4: 35 个全错 group × n5=2 = **70** 个样本被排除，总 72 个样本 ✓

这些被排除的样本的 `response_mask` 和 `advantages` 被置为 0，不参与梯度计算。

### 5.7 合并更新

日志:
```
---UPDATE--- 合并阶段: ['Stage1', 'Stage2', 'Stage3', 'Stage4'], 权重: [0.2, 0.3, 0.2, 0.3]
---UPDATE--- Stage2 padded to match max lengths
---UPDATE--- 合并后batch_size=172
---UPDATE--- Actor更新完成
```

验算 batch_size:
- Stage1: 16 (8题 × n1=2)
- Stage2: 48 (24扰动 × n3=2)
- Stage3: 36 (18通过EVS的扰动 × n4=2)
- Stage4: 72 (36 paraphrase × n5=2)
- 总计: 16+48+36+72 = **172** ✓

### 5.8 Loss 合理性分析

```
actor/pg_loss:          0.00023     (小但非零, 有效梯度)
actor/pg_clipfrac:      0.007       (0.7% 的样本触发了 PPO clip)
actor/ppo_kl:           0.008       (actor 与 initial policy 的 KL, 合理范围)
actor/grad_norm:        0.144       (梯度范数合理)
```

pg_loss 很小的原因: 大量样本被 training_mask 排除 (T2排除12/48, T4排除70/72)，实际参与梯度计算的有效样本较少。但 loss 非零说明 mask 机制正确工作——被 mask 的样本确实不贡献梯度，未被 mask 的样本正常参与更新。

Step 2 因 batch_size 不能被 4 卡整除而报错（`AssertionError: only support equal chunk. Got size of DataProto 46 and chunk 4`），这是 verl 框架的多卡数据分片限制，与 reward 逻辑无关。

## 6. 已知限制

1. **多卡 batch 整除约束**: verl 框架要求每个阶段的 batch_size 能被 GPU 数整除。当扰动/paraphrase 数量不确定时可能违反此约束。需要在 `_compute_advantage_for_stage` 或 `_merge_and_update` 中添加 padding 到整除数的逻辑。

2. **3090 24GB 显存限制**: Qwen3-4B 在 3090 上无法同时容纳 actor + vLLM + ref model。需要 48GB 以上的 GPU（如 A6000）才能在不关闭 KL loss 的情况下运行完整 pipeline。

## 7. 修改的文件清单

| 文件 | 修改内容 |
|------|---------|
| `recipe/adar_selfplay/adar_selfplay_reward.py` | 重写 compute_stage1_reward (per-rollout)，compute_stage2_reward / compute_stage4_reward 增加 training_mask |
| `recipe/adar_selfplay/adar_selfplay_ray_trainer.py` | 重写 Stage 1-4 pipeline 逻辑，添加 training_mask 应用，添加 debug_inject_t1 模式，添加详细日志 |
| `recipe/adar_selfplay/config/adar_selfplay_trainer.yaml` | 添加 debug_inject_t1 配置项 |
| `recipe/adar_selfplay/test_reward_logic.py` | 新增：单元测试脚本 |
| `scripts/adar/test_reward_verify_0.5b_20260331.sh` | 新增：0.5B 集成测试脚本 |
| `scripts/adar/test_reward_verify_4b_3090_20260401.sh` | 新增：4B 集成测试脚本 |
| `data/raw/test_8.json` | 新增：8 道测试题 |
| `data/selfplay/train_test_8.parquet` | 新增：测试数据 parquet |
