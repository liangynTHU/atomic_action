# Atomic Episode Segmentation

面向 RoboTwin 双臂轨迹的、可解释且低成本的 **state-only kinematic atomic motion segmentation** 项目。

当前主线已经从 V5 继续迭代到 V8：

```text
V5  strong events + persistent phases（按帧阈值）
 ↓
V6  timestamp-normalized state evidence
 ↓
V7  successor-trend-onset boundary
 ↓
V8  factorized left/right primitives + coordination timeline
```

**V8 是当前推荐版本。** V1–V7 保留为 baseline、消融或中间演进版本。

---

## 1. 决策输入边界

V6–V8 的切分决策严格只使用：

```text
observation.state
├── left:  xyz + quaternion + gripper
└── right: xyz + quaternion + gripper

timestamp
```

以下内容**不参与边界决策**：

- image / video；
- `action_config`；
- task text / instruction；
- object label；
- data config 中的任务语义。

视觉或 data config 可以在切分完成、边界锁定以后，用于：

- caption；
- instruction；
- object / target；
- grasp / place / operate / handover 等上层语义；
- gripper increase/decrease 到 open/close 的语义映射。

但这些后处理信息不能反向移动 V6–V8 边界。JSON 中会显式保存：

```text
selection.decision_input_policy
```

---

## 2. 方法定位

本项目区分三层 atomicity：

1. **Geometric primitive**：轨迹能否由简单 piecewise geodesic 近似；
2. **Kinematic atomic primitive**：片段内是否具有相对稳定的 move / turn / lift / lower / gripper trend；
3. **Semantic action**：包含对象、接触和任务语义，如 grasp bottle、place cup、handover。

本项目的切分器负责第 2 层。仅依赖 state 和 timestamp，不能可靠恢复完整第 3 层。

推荐的后续数据流是：

```text
state-only V8 boundaries
    → freeze boundaries
    → optional visual/data-config annotation
    → caption / instruction / semantic operation
```

---

## 3. V1–V8 演进

| 版本 | 当前命名 | 方法角色 |
|---|---|---|
| V1 | `piecewise_geodesic_approximation` | Piecewise-geodesic reconstruction baseline |
| V2 | `geometry_aware_reconstruction` | SO(3)、夹爪、Huber loss reconstruction baseline |
| V3 | `motion_change_point_ablation` | derivative-domain change-point ablation |
| V4 | `strong_event_segmentation` | pause / gripper stabilization strong-event ablation |
| V5 | `event_phase_atomic_motion` | strong event + persistent phase，仍含固定帧尺度 |
| **V6** | **`time_normalized_state_evidence`** | **所有持续时间由 timestamp 换算，边界决策明确 state-only** |
| **V7** | **`successor_onset_state_transitions`** | **把 phase crossing 修正为后继动作趋势起点** |
| **V8** | **`factorized_bimanual_state_motion`** | **左右手独立 primitive 流 + 派生 coordination timeline** |

---

## 4. V6：物理时间归一化

V5 的不少阈值隐含约 50 Hz。V6 改为先从 timestamp 估计采样周期，再把：

- minimum segment；
- pause duration；
- phase window；
- phase persistence；
- evidence merge radius；
- gripper 前后搜索窗口；

全部从秒转换为当前 episode 的帧数。

速度使用物理量：

```text
translation speed: m/s
angular speed: rad/s
gripper rate: normalized amplitude/s
```

夹爪幅值会按 episode、按手做 robust amplitude normalization，但**不会仅凭 state 猜测高值究竟表示 open 还是 closed**。因此 V6–V8 的切分证据使用：

```text
gripper_increase
gripper_decrease
```

而不是把极性假设混入边界决策。

---

## 5. V7：后继动作趋势起点原则

考虑两个动作分量重叠的情况：

```text
旧动作分数：逐渐下降
新动作分数：逐渐上升
```

不能把两条曲线的交点、旧动作完全结束点或 centered-window label 翻转点直接当边界。

V7 使用的原则是：

> 只要新动作已经出现可持续、可确认的上升趋势，边界放在新动作趋势的最早稳定起点。

若新动作为动作 1、旧动作为动作 2，则：

```text
boundary = onset(action 1)
```

而不是：

```text
boundary = crossing(action 1, action 2)
boundary = end(action 2)
```

实现中会在 persistent phase 边界附近搜索，并记录：

- `original_frame`；
- `frame`；
- predecessor score before/after；
- successor score before/after；
- `previous_trend_falling`；
- `successor_trend_rising`；
- `competing_trends_detected`；
- `boundary_policy = earliest_persistent_successor_trend_onset`。

---

## 6. V8：左右手如何划分

V5/V6/V7 的 joint fusion 可能把时间上接近的左右手证据直接合并，导致：

- 左手先启动、右手后启动时丢掉一个边界；
- 两手在同一时段做不同 phase 时被压成一个标签；
- 依次抓取被误当成 simultaneous dual grasp。

V8 改为两层输出。

### 6.1 Primary：每只手独立 primitive timeline

左右手分别完成：

```text
state evidence
→ successor-onset refinement
→ same-arm merge/filter
→ per-arm primitive timeline
```

JSON 路径：

```text
selection.per_arm_boundaries.left
selection.per_arm_boundaries.right
selection.per_arm_boundary_evidence
selection.arm_timelines.left
selection.arm_timelines.right
```

这两条 timeline 是 V8 的 primary result。

### 6.2 Derived：双臂 coordination timeline

随后才从左右手 timeline 派生统一切片：

- 同步且兼容的 phase/gripper 事件可以共享 joint boundary；
- 时间错开的事件保留为不同边界；
- 同时发生但 phase 不同的两手动作不强行改成同一动作；
- joint segment 保存 `left_primitive_id` 和 `right_primitive_id`；
- segment relation 为：
  - `left_only`；
  - `right_only`；
  - `dual_same_phase`；
  - `dual_different_phase`；
  - `both_still`。

### 6.3 双手 gripper 关系

state-only 层只做可验证的运动关系判断：

- 同方向且近同步：`synchronous_dual_gripper_event`；
- 同方向但明显错开：`sequential_same_direction_dual_gripper_events`；
- 两手方向相反：`opposing_gripper_transition_candidate`。

最后一种可以作为 handover / exchange 的候选，但仅凭 state 和未知 gripper 极性，不直接宣称 donor / recipient。视觉、接触信息或 data config 可在后处理标注该语义。

---

## 7. 数据与区间定义

用户提供的 joint-space root：

```text
/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0
```

程序会解析到配对的 16D end-pose 数据：

```text
/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/data/robotwin2.0-endpose
```

每臂：

```text
xyz + quaternion(wxyz) + continuous gripper
```

统一采用半开区间：

```text
segment = [start_boundary, end_boundary)
frame id = 0 ... T-1
boundary = 0 ... T
```

JSON 兼容历史格式，仍保存 inclusive：

```text
start_frame
end_frame
```

其中下一个边界为 `end_frame + 1`。

---

## 8. 运行

### 推荐：只跑 V6–V8

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation

python run_experiments.py \
  --data-root /apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0 \
  --episodes 0,550,6100 \
  --versions 6,7,8 \
  --vertical-direction 0,0,1 \
  --min-segment-seconds 0.10 \
  --phase-window-seconds 0.18 \
  --phase-min-seconds 0.10 \
  --onset-search-seconds 0.18 \
  --cross-arm-sync-seconds 0.08 \
  --output-dir outputs
```

### 全部消融

```bash
python run_experiments.py --versions 1,2,3,4,5,6,7,8
```

### 测试

```bash
python -m unittest discover -s tests -v
```

---

## 9. 如何看 SVG 可视化

每个 SVG 从上到下为：

1. 左手 xyz；
2. 右手 xyz；
3. 左右手 raw gripper；
4. 左右手 state motion energy；
5. 左右手独立 phase lane；
6. joint segment 标签。

颜色：

- 蓝线：左手信号；
- 橙线：右手信号；
- 红色虚线 `S`：strong state event；
- 紫色虚线 `P`：persistent phase boundary；
- 蓝色短虚线 `O`：V7/V8 successor-onset-refined boundary；
- 绿色实线：已有 weak reference，只用于对照，不参与算法。

边界标记示例：

```text
B3:L-O
```

表示：

- 第 3 个输出边界；
- 来源为 left arm；
- 由 onset policy 放置。

SVG 底部的 `Boundary audit key` 给出简写；完整证据请查看同名 JSON：

```text
selection.boundary_evidence
selection.per_arm_boundary_evidence       # V8
selection.diagnostics.onset_policy
selection.diagnostics.bimanual_policy     # V8
```

### 人工检查重点

1. 红线是否落在 gripper/pause 稳定事件附近；
2. 蓝色 `O` 是否比 phase label crossing 更早，并落在新趋势实际起点；
3. 左右手 phase lane 不同时，V8 是否仍保留各自 primitive；
4. sequential 双手事件是否被错误合并；
5. 很短 joint segment 是否来自真实错峰，还是阈值过小；
6. weak reference 只作参考，不应为了贴合绿线而改 state-only 规则。

---

## 10. 文档

- 历史数学定义与 V1–V5：[`ITERATION_REPORT.md`](ITERATION_REPORT.md)
- 本轮 V6–V8 设计、推理和使用说明：[`V6_V8_ITERATION_new.md`](V6_V8_ITERATION_new.md)

