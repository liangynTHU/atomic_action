# Atomic Episode Segmentation：V1–V29 完整迭代里程碑

## 0. 文档目的

本文是 2026 年 9 月 14 日仓库收敛重构后的**唯一历史版本文档**，完整记录
V1–V29 的研究演进、每轮主要问题、核心改动、验证结论和最终去留。

需要区分两件事：

1. V1–V29 是历史研究里程碑；
2. 当前生产代码不再暴露 `version` 参数，也不再用版本号命名模块、脚本、schema
   或测试。

当前最终能力只有：

```text
状态/时间戳 → atomic segmentation
atomic segmentation + action → action-only data
成功 episode + 当前/历史观察 → embedded CoT data
单 episode → 同一套 CoT pipeline
```

当前范围只包括**完全正确的成功演示**。V11–V13、V22–V27 中涉及的 failure、
drift、policy deviation 和 recovery 研究仍在本文留档，但其运行代码已从当前仓库
删除。

---

## 1. 总览

| 里程碑 | 核心主题 | 主要产出 | 当前去留 |
|---|---|---|---|
| V1 | 分段几何基线 | 固定段数的 piecewise-geodesic approximation | 仅历史 |
| V2 | 几何重建代价 | 位置、旋转、夹爪联合 reconstruction cost | 仅历史 |
| V3 | 运动变化点 | velocity-change ablation | 仅历史 |
| V4 | 强事件边界 | pause/gripper strong evidence | 仅历史 |
| V5 | 事件与 phase 融合 | strong/soft evidence + temporal merge | 被最终算法吸收 |
| V6 | 时间归一化 | state-only、物理时间阈值 | 被最终算法吸收 |
| V7 | successor onset | 从趋势起点而非 crossover 切分 | 被最终算法吸收 |
| V8 | 双臂因子化 | 独立 arm timeline + derived coordination | 当前分割核心 |
| V9 | canonical planner data | endpoint、guide action、chunk 合同 | 被当前 action data 吸收 |
| V10 | thinking/no-thinking | 固定 target 上生成 reasoning 变体 | 被当前 CoT 输出吸收 |
| V11 | causal evidence | pre/post/privileged evidence buckets | failure 历史 |
| V12 | failure/recovery | first break、failure chain、recovery | 已移除 |
| V13 | verifier/visual audit | temporal firewall、可视化审计 | 部分原则保留 |
| V14 | blind next action | 初始 embodied CoT 预测 | 仅历史 |
| V15 | teacher correction | future/effect/target 分层修正 | 仅审计历史 |
| V16 | answer-embedded CoT | CoT 嵌入回答并保持原 action | 格式原则保留 |
| V17 | physical grounding | 删除无证据 contact/grasp 断言 | 原则保留 |
| V18 | future-independent | 不看 future image、不复述数值 | 原则保留 |
| V19 | positive calibration | 禁止无证据肯定交互状态 | 原则保留 |
| V20 | symmetric calibration | 肯定/否定交互状态都需证据 | 原则保留 |
| V21 | provenance review | blind dual-call、target/decoy probe | 审计结论保留 |
| V22 | action-free skeleton | 先证据骨架、后独立动作决策 | 研究历史 |
| V23 | omniscient teacher | 后验教师与安全 pre-action projection | 研究历史 |
| V24 | dual critics/gate | 双 critic 和训练资格分层 | 质量原则保留 |
| V25 | object grounding | instruction provenance、object state | 质量原则保留 |
| V26 | supervised action description | action-conditioned natural reasoning | 当前生成定位 |
| V27 | strict public release | 严格公开文本重写与 VLA 去 CoT | 当前发布原则 |
| V28 | complete episode | 完整 episode 容器、逐 block CoT | 当前 episode 结构 |
| V29 | hierarchical subtask | ABC subtask → atomic actions | 数据源专项历史 |

---

# 第一阶段：从几何分段到最终双臂状态分段

## 2. V1：Piecewise-geodesic trajectory approximation

### 问题

最初需要把一条连续的双臂末端轨迹压缩成少量几何 primitive，但尚未定义
“atomic action”应由什么事件决定。

### 改动

- 把轨迹视为位置、四元数和夹爪状态的离散序列；
- 在固定段数条件下做 piecewise-geodesic approximation；
- 相邻几何 primitive 共享 knot；
- 通过动态规划最小化分段重建代价。

### 结论

V1 能给出可重复的几何压缩基线，但：

- 段数依赖外部指定；
- 几何误差不等于动作语义变化；
- pause、gripper 和双臂异步事件没有被显式建模。

### 当前去留

实现已删除，数学思路仅作为历史基线。

## 3. V2：Geometry-aware reconstruction

### 问题

V1 的统一误差尺度无法公平处理米、弧度和夹爪幅度。

### 改动

- 分别定义 position、rotation、gripper characteristic scale；
- 四元数使用 SO(3) geodesic angle，而不是直接比较四元数分量；
- 使用 robust loss 降低局部异常点对总 cost 的支配；
- 明确几何 knot 与导出 frame ownership 是不同概念。

### 结论

几何重建更稳定，但仍无法说明某个 boundary 是否对应“开始下降”“夹爪闭合”或
“双臂协同切换”。

### 当前去留

SO(3) 四元数工具保留；V2 分段器删除。

## 4. V3：Motion change-point ablation

### 问题

需要确认不做全局几何 DP、只看局部运动变化是否足够。

### 改动

- 从平移、旋转和夹爪一阶变化构造 velocity feature；
- 对特征做平滑；
- 以相邻窗口的速度差作为 change score；
- 选取高分变化点并做最小间隔合并。

### 结论

V3 对明显速度突变敏感，但容易：

- 在噪声中产生过分段；
- 把同一连续动作中的速度变化误当新动作；
- 漏掉低速但语义重要的 gripper/pause 边界。

### 当前去留

作为消融结论留档，代码删除。

## 5. V4：Strong event evidence

### 问题

单纯 change point 不稳定，需要更可信的边界证据。

### 改动

- 引入 motion-energy valley；
- 引入 gripper change 前后的 stabilization；
- 把 pause/gripper 事件定义为 strong evidence；
- 使用 temporal non-maximum suppression 合并邻近证据；
- strong evidence 在冲突时优先于普通运动变化。

### 结论

V4 边界更可解释，但只依赖强事件会漏掉：

- move → turn；
- move → lift；
- lift → fine adjustment；
- 没有明显 pause 的连续 phase transition。

### 当前去留

strong evidence 思路被最终算法保留；独立 V4 入口删除。

## 6. V5：Event- and phase-aware atomic motion

### 问题

需要在强事件之外表示持续运动 phase 的转换。

### 改动

- 根据窗口内净位移、路径长度、累计旋转和竖直比例分类
  `still/move/turn/lift/lower`；
- 对逐帧 phase 做 persistence filtering；
- 将 persistent phase transition 作为 soft evidence；
- 融合 strong/soft evidence，并保留来源、强度和双臂支持信息；
- 输出连续、不重叠的 frame slices。

### 结论

V5 建立了最终算法的基本结构：

```text
strong event evidence
+ persistent phase transition
→ temporal filtering/merging
→ atomic segments
```

问题仍是所有时间阈值以 frame 数表示，跨帧率不稳定。

### 当前去留

phase、evidence fusion 和 segment label 思路被最终算法吸收。

## 7. V6：Time-normalized state evidence

### 问题

同样的 `5 frames` 在 10 Hz、50 Hz、100 Hz 下代表不同物理时长；而且视觉和
action metadata 不应影响运动边界。

### 改动

- 所有核心阈值改为秒，再通过时间戳换算 frame 数；
- 速度改为位置/旋转/夹爪变化除以真实 `dt`；
- 明确 boundary decision 只允许读取 16D state 和 timestamp；
- image、video、task、action configuration 只能作为下游标注上下文；
- gripper 首先使用 increase/decrease 方向，不在缺少极性元数据时强行解释
  open/close。

### 验证

同一物理窗口在不同帧率下得到相应的 frame 宽度；state-only policy 被单元测试锁定。

### 当前去留

完整保留在当前 `segment_episode`。

## 8. V7：Successor-trend-onset boundary

### 问题

两个 phase 常存在重叠：旧动作尚未完全衰减，新动作已经开始。若在 score crossover
或旧动作结束处切分，boundary 会系统性偏晚。

### 改动

- 在 persistent phase proposal 周围搜索；
- 要求 successor score 具有低阈值起点和持续未来支持；
- 同时记录 predecessor falling 与 successor rising；
- boundary 选最早持续 successor onset；
- 没有可靠 onset 时保留原 proposal。

### 验证

构造 previous trend 下降、successor trend 上升的序列，确认 refined boundary 早于
nominal crossover。

### 当前去留

完整保留在当前 `successor_trend_onset` 和 `segment_episode`。

## 9. V8：Factorized bimanual state segmentation

### 问题

将双臂压成一个 global active arm 会丢失：

- 左右臂错峰动作；
- 同步但不同 phase；
- 一臂 gripper、一臂移动；
- 顺序式双臂操作。

### 改动

- 左右臂分别生成 strong/soft evidence；
- 分别做 onset refinement 和边界过滤；
- 生成独立 `left` / `right` primitive timeline；
- 再根据时间邻近和 evidence compatibility 派生 coordination timeline；
- 相同 phase 的同步事件可共享 boundary；
- 不同 phase 的错峰事件在时长允许时保持分离；
- joint segment 记录 `left_primitive_id`、`right_primitive_id` 和
  `coordination_relation`。

### 验证

在三条正常演示上得到：

```text
episode 0   : 8 segments, boundaries 25/35/49/71/88/112/127
episode 18  : 8 segments, boundaries 24/34/47/70/87/114/122
episode 780 : 6 segments, boundaries 14/35/49/71/102
```

这些边界与原最终标注逐项一致。

### 当前去留

V8 的方法就是当前生产分割算法。重构后不再称为 V8，而是：

```python
segment_episode(states, timestamps)
```

---

# 第二阶段：从 atomic segments 到 planner/action 数据

## 10. V9：Canonical planner data

### 问题

分段结果还不是可直接训练的 planner 样本，需要稳定定义：

- 当前观测；
- 合法历史；
- endpoint；
- guide action；
- chunk 数；
- 数据选择和可复算 ID。

### 改动

- 将 V8 boundary/semantic label 视为冻结上游；
- 定义 inclusive action endpoint：

```text
endpoint = action[end_frame]
```

- 定义：

```text
guide_action = action[end_frame] - observation.state[current_frame]
num_chunks = ceil((end_frame - current_frame + 1) / 8)
```

- 当前三相机图像和当前 state 可见；
- 历史样本只允许使用严格早于当前时刻的图像/state/command；
- endpoint、future state/image 和数值答案放入 hidden supervision；
- 使用稳定 hash 选择固定配额样本。

### 验证

建立 canonical row 强校验、parquet 数值复算、图片帧合法性和无未来泄漏检查。

### 当前去留

大规模 selection、分布式 shard 和部署代码删除；endpoint、guide action 和 chunk
合同保留在当前 action data generator。

## 11. V10：Thinking / no-thinking paired data

### 问题

需要比较带推理和不带推理的训练格式，同时保证两者 action target 完全一致。

### 改动

- 以 canonical target 为不可变答案；
- teacher 只为固定左右臂动作生成 rationale；
- `thinking` 输出 `<think>...</think>` 后接 canonical JSON；
- `nothinking` 直接输出同一 canonical JSON；
- 两种格式按逻辑 sample 配对，不重复改变监督权重；
- 禁止 teacher 改写 guide action、num chunks 或 segment semantics。

### 结论

明确了“reasoning 是已固定动作的监督解释”，不等同于模型独立发现正确动作。

### 当前去留

成对 ablation 和集群脚本删除；当前 CoT 行继续采用嵌入式 `<think>`，action-only
数据作为独立纯动作发布文件。

---

# 第三阶段：Failure / recovery 因果研究

## 12. V11：Structured causal evidence

### 问题

普通 caption 无法表达“当前可见事实、预期变化、实际结果、特权信息”的来源边界。

### 改动

- 建立 `pre_action`、`post_action`、`privileged` evidence bucket；
- 表达 history、precondition、blocking condition、required state change、
  expected effect 和 future affordance；
- 区分 action intent 与真实 physical effect；
- object/contact 不可见时标记 `not_observable`；
- 在短 failure episode 中发现并修复 moving-average 输出长度问题。

### 当前去留

短轨迹修复保留；failure evidence schema 和 adapter 已删除。

## 13. V12：Failure, first break and recovery

### 问题

需要在 drift 和 policy-deviation 数据中定位有证据支持的 first causal break，而不是
自由生成失败原因。

### 改动

- drift 使用 intervention window、actor trace、replan event 和 outcome；
- policy-deviation 使用 expert/applied/actual control trace；
- 区分失败 prefix 和后续 recovery rollout 的 timeline space；
- 输出 expected/observed/mismatch/first break/certainty/recovery；
- 没有证据时使用 unknown；
- 固化 provenance snapshot，避免外部 sidecar 被覆盖。

### 验证

历史冻结数据曾处理 3,156 episodes、60,175 segments，无处理错误。

### 当前去留

当前项目明确不处理 recovery，相关代码、schema、文档和示例已全部删除。

## 14. V13：Temporal verifier and visual audit

### 问题

即使 schema 区分 pre/post，生成 prompt 仍可能误混未来信息，需要代码级 firewall。

### 改动

- pre-action teacher 只接收 history/current/intent；
- post-action 才允许 observed effect、outcome、failure 和 privileged evidence；
- verifier 检查 grounding、future leakage、action/effect、mismatch、
  diagnosis support 和 uncertainty；
- 建立 HTML storyboard、因果 timeline、轨迹和关键帧审计。

### 验证

历史规则 verifier 对 60,175 segments 全部通过结构检查，并完成代表样本的多模态
模型审计。

### 当前去留

failure verifier 删除；“训练输入只含当前/历史、未来不可见”和可视化审计原则保留。

---

# 第四阶段：Embodied CoT 语义和证据约束

## 15. V14：Initial blind next-action prediction

### 问题

测试模型只根据当前/历史观察能否自行预测下一 atomic action。

### 改动

输入包括任务、前两步 action、前一/当前 state、前一/当前 head 和当前 wrist 图，
冻结 action 对初始模型隐藏。

### 验证

18 条 pilot 中 exact action match 为 6/18（33.33%）。

### 结论

视觉上合理的 reasoning 不保证动作标签正确，不能直接作为目标数据。

## 16. V15：Teacher-only future correction

### 问题

V14 的错误需要修正，但不能把未来证据写入最终训练输入。

### 改动

错误样本按以下层级校正：

```text
local future images + measured transition
→ dominant effect class
→ frozen action fallback
```

历史 18 条中最终来源为：

```text
initial 6
future observation 5
effect class 3
exact fallback 4
```

### 当前去留

只作为 teacher/audit 历史。当前成功数据生成不会读取 future image。

## 17. V16：Answer-embedded CoT

### 问题

需要同时保留原动作字段和自然语言推理，且适配常见 VLM/VLA 容器。

### 改动

- 保持冻结 segment/action 字段；
- 增加 Question、Text Reasoning Trace、Final Answer；
- CoT 可嵌入 assistant response；
- 18/18 action 与上游一致。

### 审计发现

12/18 行含无证据的 grasp/contact/force 风险。

### 当前去留

“CoT 嵌入 assistant，action-only 单独发布”的格式原则保留。

## 18. V17：Grounded physical semantics

### 问题

语言模型容易把夹爪闭合、接近物体或 action intent 写成已发生的物理效果。

### 改动

禁止无传感器依据的：

- force/torque；
- object already held；
- contact already established；
- gripper value proves grasp；
- action guarantees success。

### 验证

历史 18 条全部重写并通过 grounding 检查。

### 当前去留

保守物理表述原则保留。

## 19. V18：Future-independent, no-number distillation

### 问题

即使最终文本不直接引用 future，模型仍可能从 future image 或精确数值复制结果。

### 改动

- 只提供 previous/current image 和 state；
- 不提供 future image；
- 最终 prose 禁止复述精确数值；
- 固定 action 作为监督条件，不作为自然语言理由。

### 验证

18/18 接受，future image supplied=0。

### 当前去留

当前 CoT 生成仅使用当前决策点和可选历史决策点。

## 20. V19：Positive epistemic calibration

### 问题

“靠近 + 夹爪闭合”仍常被表述为 confirmed grasp。

### 改动

禁止无证据的正向 interaction claim；允许：

```text
prepare for a possible grasp
physical interaction remains unverified
attempt to establish contact
```

### 验证

历史 V18 中 7 条风险样本被修正，18/18 接受。

### 当前去留

当前模板和模型 prompt 都要求不把 commanded motion 写成已完成效果。

## 21. V20：Symmetric epistemic calibration

### 问题

缺乏证据时，不仅不能说“已经抓住”，也不能说“确定没有抓住”。

### 改动

建立对称规则：

```text
unsupported positive claim → reject
unsupported negative claim → reject
unobservable state → uncertain or omit
```

### 验证

历史 18 条中发现 2 条负向过度断言并重新生成，最终 18/18 接受。

### 当前去留

对称 uncertainty 原则保留。

## 22. V21：Reasoning provenance review

### 问题

action-conditioned rationale 不能被误称为模型原生、独立、忠实的内部思维过程。

### 改动

- 新建 blind view，物理排除 hidden target/future/teacher answer；
- Generator 和 Reviewer 独立 blind call；
- target mismatch 不允许触发语义重试；
- raw hidden reasoning 只进 private audit；
- 增加 true-target 与 decoy-target probe；
- 训练资格区分 independent policy CoT 与 action-conditioned explanation。

### 结论

绝大多数历史 CoT 应被定性为：

```text
action-conditioned supervised explanation
```

而不是 independent policy reasoning。

### 当前去留

当前 schema 明确写入 `action_conditioned=true`，不做忠实思维声明。

---

# 第五阶段：高质量门控与监督数据发布

## 23. V22：Action-free evidence skeleton

### 问题

要检验 independent reasoning，必须先构造不含 action target 的证据骨架，再选动作。

### 改动

- pre-action facts；
- task progress；
- missing condition；
- required state change；
- why now；
- future affordance；
- uncertainties；
- Generator/Reviewer 各自冻结 skeleton 后才选择 action；
- commit 后才与参考动作比较；
- mismatch 不重试。

### 验证

9 条代表样本：

```text
generator target match 6/9
reviewer target match 6/9
action consensus 7/9
strict generator 1/9
strict reviewer 2/9
```

### 当前去留

作为 independent-CoT 研究结论留档；当前生产目标是 supervised
action-conditioned CoT。

## 24. V23：Omniscient teacher and sanitized projection

### 问题

后验教师可提高事实审计能力，但未来、outcome 和 privileged evidence 不能进入
pre-action 训练文本。

### 改动

教师输出分为：

```text
reference action assessment
pre-action projection
post-action validation
omniscient explanation
```

只有 sanitized pre-action projection 可进入后续训练生成。

### 验证

历史 9/9 teacher rows 通过 critic；其中 7 条模型生成、2 条保守 fallback。

### 当前去留

omniscient/failure pipeline 删除；pre-action projection 的隔离原则保留。

## 25. V24：Dual critics and quality gate

### 问题

单次生成与单 critic 不足以防止循环论证、future leakage 和 unsupported effect。

### 改动

- 最终训练文本只读取 blind packet、blind result 和 sanitized projection；
- 两个 target-blind critic 独立审查；
- mandatory gate 后再分 Gold/Silver/Reject；
- 公开数据不包含 private hidden reasoning。

### 验证

9 条代表样本：

```text
candidate training rows 3
strict training rows 1
independent policy CoT 0
```

项目选择保留真实拒绝，而不是放宽标准制造数量。

### 当前去留

复杂双 critic 代码删除；target isolation、保守表达和可审计 manifest 保留。

## 26. V25：Instruction provenance and object grounding

### 问题

需要区分真实 rollout prompt 与后验恢复的 task semantics，并补足 object-centric
pre-action state。

### 改动

- instruction resolution：
  `exact_recorded_instruction`、`exact_provenance_prompt`、
  `canonical_task_semantics`、`candidate_paraphrase_unconfirmed`；
- target-blind object observer；
- entity、arm-object、support、interaction relation；
- 每个关系记录 visibility、certainty 和 evidence ID；
- 禁止 `closed gripper = confirmed grasp`、`proximity = contact`；
- object state 双 critic；
- 扩展质量等级和 VLM/VLA 导出。

### 当前去留

失败数据特有 provenance 和 object observer 删除；成功 manifest 的
`success_provenance` 与 conservative reasoning 规则保留。

## 27. V26：Supervised action-description CoT

### 问题

用户需要稳定、与真实 action 对齐的 CoT，而不是低 exact-match 的独立动作预测。

### 改动

- 模型可见 frozen supervised transition；
- 仍禁止 future image、post-action state、outcome 和 privileged evidence；
- 输出自然的 state-to-action reasoning 和双臂 action description；
- 双 critic 检查 grounding、non-circular、action fidelity、future leakage、
  unsupported effect 和 conciseness；
- 失败时有 bounded rewrite 和保守 fallback。

### 当前去留

这成为当前 CoT 的正式定位：

```text
action-conditioned supervised pre-action explanation
```

## 28. V27：Strict public rewrite and release

### 问题

需要将研究审计文本收敛成可以公开训练的 VLM 数据，并确保 VLA 完全不含 CoT。

### 改动

- 严格移除 contact/grasp/force/torque/outcome 等无证据表述；
- target 与当前 state/segment kinematics 矛盾时 quarantine；
- VLM assistant 内嵌 `<think>`；
- VLA 只保留 task/segment/action/boundary/guide action/chunks；
- 禁止 VLA 出现 reasoning、CoT、assistant response、provenance chain。

### 验证

历史 30 条输入中 28 条训练可用、2 条因 target/当前状态矛盾被隔离。

### 当前去留

当前发布继续严格分离：

```text
action_train.json   → 无 CoT
cot_train.jsonl     → assistant 内嵌 CoT
```

当前只接收成功清单，因此不再需要 failure quarantine 分支。

---

# 第六阶段：完整 episode 与层级任务

## 29. V28：Complete episode coverage

### 问题

早期 pilot 多为零散 segment，无法审计完整成功 episode 的动作覆盖和历史连续性。

### 改动

- one episode 作为完整 grouping/audit container；
- one atomic block 仍是一条 policy training row；
- 当前 block 可使用当前三相机、当前 state、已完成 action history；
- 非初始 block 可使用上一 boundary 三相机和 state；
- 当前 end image/state、future block、episode outcome 不进入训练 human；
- 完整视频只用于人工审计；
- VLM 有 embedded CoT，VLA 纯 action。

### 历史验证

历史 pilot 曾包含 4 条完整 episode、61 个 block，其中成功与 failure/recovery
混合。当前重构只继承正常成功 episode 的结构，不保留 failure 两条。

### 当前去留

完整保留为当前 episode 生成和完整可视化结构：

```text
one successful episode
→ all atomic segments
→ all CoT rows
→ three complete camera videos
→ segment-by-segment audit page
```

## 30. V29：ABC-130K subtask to atomic actions

### 问题

探索另一类数据源：完整 instruction 下已有 semantic subtask，需要进一步细分为
atomic action。

### 改动

- 使用 ABC-130K 的 episode → subtask 层级；
- subtask boundary 作为 proposal，在原 joint/gripper stream 附近局部调整；
- 每个 subtask 再拆 1–4 个 atomic actions；
- 生成 VLM+CoT 和 action-only VLA；
- 保存 boundary adjustment provenance；
- 三任务 pilot 使用 68 subtasks、115 atomic actions；
- 历史审计记录 115/115 critic PASS。

### 结论

V29 证明了同一发布思想可以用于层级任务数据，但其：

- 14D joint-state；
- MCAP 解码；
- ABC-specific hierarchy；
- source boundary refinement；

都属于另一数据源的专项转换逻辑，不应和当前 16D RoboTwin end-pose 成功演示核心
混在一起。

### 当前去留

V29 专项代码、schema、下载和验证脚本删除；经验留在本文。若未来正式支持
ABC-130K，应建立独立 adapter/package。

---

# 第七阶段：2026-09-14 仓库收敛

## 31. 为什么停止继续增加版本号

V1–V29 混合了四类不同概念：

1. segmentation 算法迭代；
2. 数据格式迭代；
3. CoT 生成和质量审计；
4. failure/recovery 与不同数据源的专项研究。

继续增加编号会造成：

- 用户不知道应该运行哪一个脚本；
- 同一能力存在多个近似实现；
- 文件名和 schema 被历史版本号绑定；
- failure/recovery 代码可能误进入成功训练数据；
- 文档、部署脚本和本地绝对路径大量重复；
- 测试验证历史实现，而不是最终产品合同。

因此本轮重构不再创建“V30”，而是建立稳定功能 API。

## 32. 最终保留的算法和数据合同

### 分割

```python
segment_episode(states, timestamps)
```

等价于历史最终 V8 方法，但代码和输出不再出现版本号。

### Action-only

```text
successful episode
→ segmentation
→ action[end] - state[start]
→ pure action episode JSON
```

### CoT

```text
current/history three-camera observations
+ current/history state
+ completed action history
+ supervised current atomic target（仅生成阶段可见）
→ conservative pre-action explanation
→ <think>...</think> + action JSON
```

最终训练 human 中不包含当前 target。

### 单 episode

单 episode 与批量生成共用同一代码，不维护第二套数据逻辑。

### 可视化

收敛时已使用三条人工确认的正常演示完成验收：

- 3 条 segmentation SVG；
- 3 条 complete episode HTML；
- 每条 complete episode HTML 包含三路完整视频。

真实媒体和 ZIP 只保存在本地归档，不进入代码仓库。Git 中保留生成器和纯合成
测试，持续验证同样的 `3 SVG + 3 HTML + 9 MP4` 打包合同。

## 33. 删除内容

本轮删除：

- V1–V7 独立运行实现；
- 版本调度器；
- canonical selection/sharding 历史实现；
- failure/recovery adapter、teacher、verifier 和 visualization；
- 四路 ablation；
- 旧 high-quality/review/supervised/full-episode 版本模块；
- ABC-130K 专项转换；
- Qwen 集群部署和模型下载工具；
- 所有旧版本 schema；
- 旧 completion audit、handoff、部署、迭代和 recovery 文档；
- 历史 `run_experiments.py`；
- 只服务旧版本的测试。

这些内容的关键结论已集中到本文。

## 34. 最终验证

收敛时执行以下验证：

- 三条正常演示分割边界与重构前最终算法逐项一致；
- action-only 数据不含 CoT 字段；
- 非成功 manifest 被拒绝；
- CoT human 不暴露当前 guide action/sub-task；
- 三个 segmentation SVG 存在；
- 三个 complete episode HTML 存在；
- 每个 complete episode 页面包含三路完整视频；
- 所有 HTML 本地引用可解析；
- 全量测试通过；
- Git diff whitespace 检查通过。

---

## 35. 最终原则

当前仓库的长期原则是：

1. **历史版本只存在于本文。**
2. **生产代码使用功能名称，不使用研究编号。**
3. **边界只由 state 和 timestamp 决定。**
4. **action-only 与 CoT 数据严格分开。**
5. **CoT 是 action-conditioned supervised explanation，不声称是模型内部思维的忠实转录。**
6. **成功清单必须显式确认，failure/recovery 不得混入。**
7. **完整 episode 可审计，但未来帧不得泄漏到单步训练输入。**
8. **不同数据源的专项 adapter 应独立维护，不再污染核心仓库。**
