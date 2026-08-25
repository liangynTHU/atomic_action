# Atomic Episode Segmentation：V6–V8 迭代设计与使用说明

整理日期：2026-08-12

本文是本轮新增内容的主说明文档，覆盖：

- 为什么继续从 V5 迭代；
- V6、V7、V8 各自解决什么问题；
- state-only 决策约束；
- 左右手不同动作的划分；
- “新动作上升、旧动作下降”时的边界原则；
- JSON / SVG 如何检查；
- 后续 caption / instruction 如何接入但不污染切分。

---

# 0. 本轮最终结论

当前推荐使用 **V8**：

```text
V8 = timestamp-normalized state evidence
   + successor-trend-onset boundary
   + factorized left/right primitive streams
   + derived bimanual coordination timeline
```

V8 的最重要约束是：

> 切分只由 state 和 timestamp 决定。  
> 视觉、task text、action_config、data config 只能在边界冻结后用于标注。

本轮没有把 `embodied-task-relabel` 的视觉或 `action_config` 决策路径直接并入切分器，而是只吸收了其中合理的结构经验：

1. 左右手不能只保留一个 global active arm；
2. simultaneous dual 与 sequential dual 必须区分；
3. 两手可能同时做不同动作；
4. handover 至少需要 donor / recipient 关系，但 state-only 层只能安全地产生 exchange candidate；
5. outgoing action 的边界应靠“开始移动”的 onset，而不是等待旧状态完全结束。

---

# 1. 用户要求与实现对应关系

## 1.1 轨迹切割只依靠 state

实现：

```text
selection.decision_input_policy.boundary_decision_inputs
```

只包含：

- left/right xyz；
- left/right quaternion；
- left/right gripper；
- timestamp。

明确排除：

- image；
- video；
- `action_config`；
- task text；
- instruction；
- object label。

## 1.2 caption / instruction 可以看视觉或 data config

实现上采用“边界锁定”约定：

```text
state segmentation
→ boundaries frozen
→ visual/config annotation
```

视觉或 config 可以补充：

- object；
- target；
- semantic skill；
- instruction；
- gripper polarity；
- handover donor / recipient。

但不允许修改：

- `chosen_boundaries`；
- `per_arm_boundaries`；
- motion primitive ownership。

## 1.3 左右手不同内容如何划分

V8 输出：

```text
left primitive timeline
right primitive timeline
derived coordination timeline
```

左右手各自边界优先，joint timeline 只是派生视图。

## 1.4 新动作增加、旧动作减少

V7/V8 采用：

```text
boundary = earliest persistent successor onset
```

不采用：

```text
boundary = two scores crossing
boundary = previous action fully ended
boundary = centered-window label switched
```

## 1.5 多版本逐步迭代

- V6：修正时间尺度与输入契约；
- V7：修正 transition boundary 的定义；
- V8：修正双臂结构。

这三个版本不是平行堆功能，而是每一版只解决上一版暴露出的核心问题。

---

# 2. State-only 输入契约

设 episode 有 $T$ 个状态：

$$
\mathcal X
=
\left\{
\left(t_i,\boldsymbol x_i\right)
\right\}_{i=0}^{T-1},
\qquad
t_0<t_1<\cdots<t_{T-1}.
$$

每只手 $h\in\{L,R\}$：

$$
\boldsymbol x_i^h
=
\left(
\boldsymbol p_i^h,
R_i^h,
g_i^h
\right),
$$

其中：

- $\boldsymbol p_i^h\in\mathbb R^3$；
- $R_i^h\in SO(3)$；
- $g_i^h$ 是连续 gripper state。

完整状态：

$$
\boldsymbol x_i
=
\left(
\boldsymbol x_i^L,
\boldsymbol x_i^R
\right).
$$

边界决策函数严格写为：

$$
\mathcal B
=
F_{\mathrm{state}}
\left(
\left\{t_i,\boldsymbol x_i\right\}_{i=0}^{T-1}
\right).
$$

不允许变成：

$$
\mathcal B
=
F
\left(
\mathcal X,
\mathrm{video},
\mathrm{action\_config},
\mathrm{text}
\right).
$$

后者可以存在于 semantic annotation 层，但不能作为本项目的 motion segmentation 定义。

---

# 3. 统一区间定义

所有持久化 primitive 使用半开区间：

$$
[b_j,b_{j+1}).
$$

边界：

$$
0=b_0<b_1<\cdots<b_m=T.
$$

JSON 为兼容历史代码，保存：

```text
start_frame = b_j
end_frame   = b_{j+1} - 1
```

因此一个 JSON segment 的下一个边界是：

```text
end_frame + 1
```

这一点在看 SVG 和手工对照时非常重要。

---

# 4. V6：Timestamp-Normalized State Evidence

## 4.1 为什么需要 V6

V5 虽然读取 timestamp，但核心参数仍大量按帧定义：

- `phase_window=9`；
- `boundary_merge_radius=5`；
- gripper 前窗口 30 帧；
- gripper 后窗口 20 帧；
- minimum duration 5 帧。

在约 50 Hz 的 RoboTwin 上，这些参数对应：

```text
9 frames  ≈ 0.18 s
5 frames  ≈ 0.10 s
30 frames ≈ 0.60 s
20 frames ≈ 0.40 s
```

如果直接迁移到 30 Hz、100 Hz 或不规则 timestamp，方法含义会改变。

V6 的目标不是增加语义，而是先把 state-only 切分的时间基础做正确。

## 4.2 时间参数解析

估计采样周期：

$$
\Delta t
=
\operatorname{median}
\left(
t_i-t_{i-1}
\right).
$$

物理持续时间 $\tau$ 转为帧数：

$$
N_\tau
=
\max
\left(
1,
\operatorname{round}
\left(
\frac{\tau}{\Delta t}
\right)
\right).
$$

V6–V8 使用该方式解析：

- minimum segment；
- pause minimum；
- phase window；
- phase persistence；
- boundary merge；
- onset search；
- cross-arm sync；
- gripper pre/post search。

## 4.3 物理速度

位置速度：

$$
\boldsymbol v_i^h
=
\frac{
\boldsymbol p_i^h-\boldsymbol p_{i-1}^h
}{
t_i-t_{i-1}
}.
$$

角速度使用 $SO(3)$ log delta：

$$
\boldsymbol\omega_i^h
=
\frac{
\operatorname{Log}
\left(
\left(R_{i-1}^h\right)^{-1}R_i^h
\right)
}{
t_i-t_{i-1}
}.
$$

gripper rate：

$$
\dot g_i^h
=
\frac{
\widetilde g_i^h-\widetilde g_{i-1}^h
}{
t_i-t_{i-1}
}.
$$

其中 $\widetilde g$ 是 robust amplitude normalization 后的信号。

## 4.4 Gripper 极性不进入边界决策

不同数据可能满足：

```text
high = open,  low = closed
```

也可能满足：

```text
high = closed, low = open
```

如果没有可靠 metadata，仅凭一条轨迹不能总是正确判断极性。

因此 V6–V8 的运动层只保存：

```text
increase
decrease
```

以及：

- onset；
- done；
- normalized delta；
- source arm。

以后 data config 可以把 decrease 映射为 close，但该映射只是 annotation，不影响已经产生的边界。

## 4.5 State motion energy

V6 定义 dimensionless energy：

$$
E_i^h
=
\frac{\|\boldsymbol v_i^h\|}{s_v}
+
\frac{\|\boldsymbol\omega_i^h\|}{s_\omega}
+
\frac{|\dot g_i^h|}{s_g}.
$$

其中 $s_v,s_\omega,s_g$ 是 characteristic physical-rate scales。

pause / stabilization 证据来自持续低能量区间，不依赖图像中的接触判断。

## 4.6 V6 强证据与软证据

Strong：

- sustained state-motion energy valley；
- pre-gripper stabilization；
- post-gripper stabilization。

Soft：

- persistent `still/move/turn/lift/lower` phase transition。

V6 仍使用接近 V5 的 joint temporal fusion，因此它解决了：

- 时间归一化；
- state-only 输入契约；
- gripper 极性污染；

但没有完全解决：

- centered phase window 导致的边界偏移；
- 两手近邻证据误合并。

这正是 V7、V8 的动机。

---

# 5. V7：Successor-Trend-Onset Boundary

## 5.1 问题形式化

设旧动作 phase 为 $a_{\mathrm{old}}$，新动作 phase 为 $a_{\mathrm{new}}$。

对应活动分数：

$$
s_{\mathrm{old}}(t),
\qquad
s_{\mathrm{new}}(t).
$$

常见平滑控制 transition：

$$
\frac{d}{dt}s_{\mathrm{old}}(t)<0,
\qquad
\frac{d}{dt}s_{\mathrm{new}}(t)>0.
$$

如果使用 score crossing：

$$
t_{\mathrm{cross}}
=
\min
\left\{
t:
s_{\mathrm{new}}(t)
\ge
s_{\mathrm{old}}(t)
\right\},
$$

边界会偏晚，因为新动作在 crossing 前已经开始。

## 5.2 本轮采用的原则

定义新动作的 earliest persistent onset：

$$
t^\star
=
\min_t
\left\{
\begin{array}{l}
s_{\mathrm{new}}(t)\ge\theta_{\mathrm{low}},\\
\operatorname{mean}
\left(
s_{\mathrm{new}}[t:t+K]
\right)
\ge\theta_{\mathrm{sustain}},\\
\operatorname{support\_fraction}
\left(
s_{\mathrm{new}}[t:t+K]
\right)
\ge\rho,\\
s_{\mathrm{new}}\text{ 相对前窗明显上升}
\end{array}
\right\}.
$$

最终：

$$
\boxed{
b=t^\star
}
$$

而不是 $t_{\mathrm{cross}}$。

## 5.3 对用户提出情形的直接回答

若：

```text
动作1：逐渐增加
动作2：逐渐减少
```

并且动作1是后继动作，则切分原则应是：

> 以动作1出现持续上升趋势的最早时刻为边界。

动作2是否已经完全降为零，不应阻止切分。

原因：

1. 平滑控制中两个 motion component 会重叠；
2. crossing 只说明两者强弱关系反转，不等于新动作开始；
3. 等旧动作结束会把新动作前半段错误分给旧 segment；
4. 对 action prediction / chunking，后继控制模式的启动更接近因果边界。

## 5.4 Phase activity score

V7 对每个 frame 构造单侧、速度域活动分数：

- `move`：horizontal speed；
- `lift`：positive vertical speed；
- `lower`：negative vertical speed；
- `turn`：angular speed，并扣除 translation competition；
- `still`：motion energy 的反函数。

这些 score 用于定位 onset；persistent window phase label 仍用于确定 transition 类型。

因此：

```text
persistent phase label = transition proposal
causal velocity score  = boundary localization
```

而不是让单帧 score 独立决定所有边界。

## 5.5 审计信息

每个 onset-refined candidate 保存：

```json
{
  "original_frame": 117,
  "frame": 112,
  "previous_phase": "turn",
  "current_phase": "lift",
  "boundary_policy": "earliest_persistent_successor_trend_onset",
  "onset_refinement": {
    "previous_score_before": 1.2,
    "previous_score_after": 0.7,
    "successor_score_before": 0.1,
    "successor_score_after": 0.9,
    "previous_trend_falling": true,
    "successor_trend_rising": true,
    "competing_trends_detected": true
  }
}
```

数值仅为结构示例。

## 5.6 Strong event 优先级

如果 onset candidate 与明确 gripper/pause strong event 落在同一 temporal cluster：

```text
strong event > soft onset
```

V7 不用 soft phase 把明确 manipulation event 覆盖掉。

但在纯 soft cluster 内：

```text
earliest valid successor onset
```

优先于 weighted average。

---

# 6. V8：Factorized Bimanual Segmentation

## 6.1 为什么 joint union 不够

把两手证据直接放进一个 temporal NMS 有一个隐含假设：

> 时间接近的左右手变化属于同一个边界。

该假设在以下情况成立：

- 两手同步抬锅；
- 两手同步抓同一长物体；
- 两手同步进入 still。

但在以下情况不成立：

- 左手先抓、右手后抓；
- 一只手固定，另一只手旋转；
- 左手开始 lift，右手仍在 move；
- handover 中 recipient close 与 donor open 是有顺序的；
- 两手分别处理两个物体。

因此 V8 不再把 joint timeline 当唯一真值。

## 6.2 Primary representation

分别求：

$$
\mathcal B^L
=
F_{\mathrm{arm}}
\left(
\mathcal X^L
\right),
$$

$$
\mathcal B^R
=
F_{\mathrm{arm}}
\left(
\mathcal X^R
\right).
$$

对应 primitive：

$$
\mathcal P^L
=
\left\{
P_0^L,P_1^L,\ldots
\right\},
$$

$$
\mathcal P^R
=
\left\{
P_0^R,P_1^R,\ldots
\right\}.
$$

每只手独立执行：

```text
strong/soft evidence
→ successor onset
→ same-arm temporal merge
→ minimum-duration filter
→ arm primitive timeline
```

## 6.3 Derived coordination view

joint boundary set：

$$
\mathcal B^{C}
=
G
\left(
\mathcal B^L,
\mathcal B^R
\right).
$$

关键点：

> $\mathcal B^C$ 是派生结果，不能覆盖或删除 $\mathcal B^L,\mathcal B^R$ 中的 primary evidence。

### 同步兼容

若左右手边界时间差：

$$
|b^L-b^R|
\le
\tau_{\mathrm{sync}},
$$

且 evidence signature 相同，例如：

```text
left:  move -> lift
right: move -> lift
```

则可形成一个：

```text
synchronous_phase
```

coordination boundary。

### 同步但不同 phase

例如：

```text
left:  move -> lift
right: move -> turn
```

即使时间接近，也不能宣称是同一 per-arm primitive。

若时间差足以形成稳定 joint segment，则保留两个边界；若差异短到低于 coordination minimum duration，则 joint view 可以合并成：

```text
near_simultaneous_mixed_evidence
```

但左右手 primary timeline 仍保留各自边界。

### Sequential dual

如果两手同方向 gripper event 明显错开：

```text
left decrease at t1
right decrease at t2
|t2-t1| > sync threshold
```

输出：

```text
sequential_same_direction_dual_gripper_events
```

不把它自动合并成一次 simultaneous dual event。

### Opposing gripper trends

若：

```text
left increase
right decrease
```

或反之，输出：

```text
opposing_gripper_transition_candidate
```

该信号可能对应：

- handover；
- 一手放、一手抓；
- 两个无关物体的并行动作；
- gripper polarity 与预期相反。

因此 state-only 层不能直接写：

```text
donor = left
recipient = right
```

除非后处理获得：

- gripper polarity；
- visual contact；
- object flow；
- data config 语义。

## 6.4 Joint segment relation

V8 的每个 joint segment 保存：

```text
coordination_relation
left_primitive_id
right_primitive_id
```

关系定义：

| relation | 含义 |
|---|---|
| `both_still` | 两手均为 still |
| `left_only` | 左手非 still，右手 still |
| `right_only` | 右手非 still，左手 still |
| `dual_same_phase` | 两手均在动且 phase 相同 |
| `dual_different_phase` | 两手均在动但 phase 不同 |

这比单个：

```text
label = left_action + right_action
```

更适合后续检索和训练数据分析。

---

# 7. V6–V8 输出结构

## 7.1 公共字段

```text
selection.decision_input_policy
selection.temporal_parameters
selection.strong_candidates
selection.soft_candidates
selection.boundary_evidence
selection.chosen_boundaries
selection.diagnostics
```

## 7.2 V7 onset 字段

```text
boundary_policy
original_frame
onset_refinement
```

## 7.3 V8 双臂字段

```text
selection.per_arm_boundaries
selection.per_arm_boundary_evidence
selection.arm_timelines
selection.coordination_boundaries
selection.coordination_boundary_evidence
selection.diagnostics.bimanual_policy
selection.diagnostics.synchronous_dual_gripper_events
selection.diagnostics.sequential_dual_gripper_events
selection.diagnostics.cross_arm_exchange_candidates
```

joint segment：

```text
left_action
right_action
coordination_relation
left_primitive_id
right_primitive_id
```

---

# 8. 如何看可视化

SVG 的目标不是“漂亮地画轨迹”，而是让每个边界可审计。

## 8.1 面板

从上到下：

1. Left xyz；
2. Right xyz；
3. left/right raw gripper；
4. left/right state motion energy；
5. left/right phase lane；
6. joint segment label。

## 8.2 线型

```text
S = strong state event
P = persistent phase boundary
O = onset-refined soft boundary
```

颜色：

- 红：strong；
- 紫：persistent phase；
- 蓝色短虚线：successor onset；
- 绿：weak reference；
- 蓝信号：left；
- 橙信号：right。

## 8.3 标记读取

```text
B4:LR-S
```

解释：

- `B4`：第 4 个输出边界；
- `LR`：左右手共同支持；
- `S`：strong。

```text
B2:L-O
```

解释：

- 第 2 个边界；
- 左手证据；
- successor-onset refined。

## 8.4 检查 V7

对 `O` 边界：

1. 找 phase lane 的 nominal transition；
2. 看 `O` 是否略早于 label crossing；
3. 看新动作速度分量是否从 `O` 附近开始持续增加；
4. 看旧动作分量是否同步下降；
5. 在 JSON 检查 `competing_trends_detected`。

若 `O` 早到了明显噪声区：

- 缩小 `--onset-search-seconds`；
- 增大 onset confirm duration；
- 调高 activity threshold。

若 `O` 仍落在 crossing：

- 检查 successor activity score 是否选错分量；
- 检查 phase 类型是否错误；
- 检查当前速度是否被 centered smoothing 提前/延后。

## 8.5 检查 V8

重点看两条 phase lane：

### 情况 A：两手同步

两条 lane 同时换 phase，joint boundary 应为 `LR`。

### 情况 B：两手错峰

一条 lane 先换，另一条后换：

- per-arm JSON 应保留两个边界；
- joint SVG 可以出现两个边界；
- 中间 segment 可能为 `dual_different_phase`、`left_only` 或 `right_only`。

### 情况 C：两手分别抓取

查看 gripper 两条曲线：

- 若 onset 明显分离，不应被写成一个 synchronous event；
- JSON 应出现 sequential candidate。

### 情况 D：相反 gripper 方向

JSON 中可以出现 exchange candidate，但不要仅凭此把它当 handover GT。

## 8.6 Boundary audit key

SVG 底部只放简写。完整来源始终以 JSON 为准：

```text
selection.boundary_evidence
```

每个 evidence 可以追踪到：

- source arm；
- evidence class；
- evidence type；
- support cluster；
- original/refined frame；
- phase transition；
- onset score；
- gripper interval。

---

# 9. 参数解释

## 9.1 推荐默认值

```text
min_segment_seconds        = 0.10
boundary_merge_seconds     = 0.10
phase_window_seconds       = 0.18
phase_min_seconds          = 0.10
onset_search_seconds       = 0.18
onset_confirm_seconds      = 0.06
cross_arm_sync_seconds     = 0.08
coordination_min_seconds   = 0.04
```

## 9.2 调参顺序

推荐按以下顺序：

1. 确认 timestamp 和单位；
2. 调 motion physical scales；
3. 调 phase window；
4. 调 phase minimum duration；
5. 调 onset search / confirm；
6. 最后调 cross-arm sync。

不建议先针对某一条 episode 改 fixed frame constants。

## 9.3 Cross-arm sync 的含义

`cross_arm_sync_seconds` 不是：

```text
小于该值就一定合并
```

而是：

```text
小于该值才有资格检查是否为 compatible synchronous evidence
```

phase 不同、gripper 方向相反时，仍应优先保留差异。

---

# 10. 评价建议

## 10.1 Motion boundary GT

应单独标注：

- left boundaries；
- right boundaries；
- coordination boundaries；
- phase label；
- successor onset。

不能只给一个扁平的 episode segment GT，否则无法评价 V8 的 factorized result。

## 10.2 指标

### Per-arm

- left/right boundary F1；
- onset localization error；
- phase accuracy；
- over/under segmentation；
- per-arm duration distribution。

### Coordination

- synchronous event precision/recall；
- sequential event separation accuracy；
- `left_only/right_only/dual_*` relation accuracy；
- joint short-segment rate。

### Semantic 后处理

另行评价：

- object；
- target；
- grasp/place/operate/handover；
- donor/recipient；
- instruction quality。

不要把 semantic accuracy 混入 state boundary F1。

## 10.3 当前 weak reference

项目中的 existing JSON reference 仍是 weak automatic segmentation。

因此：

```text
weak-reference boundary F1
```

只表示一致性，不代表 V8 的真实准确率。

特别是 V7 会有意把边界从 crossing 提前到 successor onset，它可能降低 weak-reference F1，却更符合本轮边界定义。

---

# 11. Caption / Instruction 接口建议

边界冻结后，每个 segment 可以构建 annotation request：

```json
{
  "segment_id": 3,
  "start_frame": 112,
  "end_frame": 126,
  "left_action": "lift",
  "right_action": "still",
  "coordination_relation": "left_only",
  "state_boundary_locked": true,
  "optional_context": {
    "video_frames": [],
    "action_config_hint": null,
    "task_text": null,
    "data_config": null
  }
}
```

annotation model 可以输出：

```json
{
  "caption": "Lift the bottle with the left arm.",
  "instruction": "Raise the grasped bottle vertically.",
  "object": "bottle",
  "semantic_skill": "lift_after_grasp"
}
```

但不能输出新的 start/end 覆盖原切分。

如果未来确实需要视觉修边，应创建独立字段：

```text
semantic_window_proposal
```

并保留：

```text
state_motion_boundary
```

两者不能静默覆盖。

---

# 12. 已实现测试

新增单元测试覆盖：

1. 50 Hz / 100 Hz 下 duration-to-frame 换算；
2. V6 决策策略排除 visual 和 action_config；
3. competing trends 下 onset 早于 crossing；
4. 同步同 phase 左右手 evidence 合并；
5. 近邻但不同 phase 的左右手 evidence 保留；
6. V8 输出独立 left/right timelines；
7. 原 V1–V5 geometry/evidence 测试继续通过。

运行：

```bash
python -m unittest discover -s tests -v
```

---

# 13. 已知限制

## 13.1 State-only 不能可靠判断接触

没有 force/tactile/vision 时，gripper change 不一定表示成功 grasp。

## 13.2 Gripper polarity 语义未自动决定

V6–V8 故意使用 increase/decrease。若数据集可靠提供：

```text
0=open, 1=closed
```

可在 annotation stage 映射。

## 13.3 Base motion

当前默认 end pose 与 world/base frame 关系稳定。移动底盘任务应额外输入 base pose，先分离：

```text
base motion
arm relative motion
```

## 13.4 Coordination timeline 不是唯一真值

V8 的 per-arm timeline 是 primary。训练下游模型时应根据目标选择：

- per-arm chunks；
- joint coordination chunks；
- 或二者映射。

## 13.5 Onset 仍依赖 phase proposal

如果 persistent phase 类型本身判错，onset score 也会沿错误 successor phase 搜索。

因此需要同时检查：

- phase classification；
- onset localization。

---

# 14. 推荐运行方式

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation

python run_experiments.py \
  --episodes 0,550,6100,10599 \
  --versions 6,7,8 \
  --vertical-direction 0,0,1 \
  --output-dir outputs
```

推荐至少包含：

- 单臂 episode；
- 同步双臂 episode；
- 两手不同 phase episode；
- sequential dual gripper episode；
- opposing gripper transition episode。

---

# 15. 版本选择

## 只想复现实验历史

使用 V1–V5。

## 想验证 timestamp 归一化

比较 V5 与 V6。

## 想验证“新动作趋势起点”

比较 V6 与 V7，并重点看 SVG 中 `O` 边界。

## 想生成当前推荐的双臂训练切片

使用 V8，并同时读取：

```text
segments
selection.arm_timelines
selection.per_arm_boundary_evidence
```

不要只读取 joint `segments` 后丢掉 per-arm timeline。

---

# 16. 最终方法声明

本轮最终方法不是视觉动作分割，也不是依赖任务配置的子任务切分。

它是：

> 基于双臂机器人 state 与 timestamp 的、物理时间归一化、后继趋势起点感知、左右手因子化的运动学原子动作切分。

视觉与 data config 的正确位置是：

> 在 state boundary 冻结以后，为 segment 生成 caption、instruction 和 semantic relation。

