# Episode 原子动作切分：统一数学建模与 V1–V8 方法报告

## 0. Scope 与核心结论

本项目研究的是：给定双臂机器人末端位姿、夹爪和时间戳，将一个 episode 划分成数量不固定的 **kinematic atomic motion segments**。

本文原始主体记录 V1–V5；2026-08-12 在文末追加 V6–V8。完整演进为：

$$
\boxed{
\text{Geometry}
\rightarrow
\text{Proper geometry}
\rightarrow
\text{Dynamics}
\rightarrow
\text{Strong events}
\rightarrow
\text{Strong events + persistent phases}
\rightarrow
\text{Physical-time state evidence}
\rightarrow
\text{Successor onset}
\rightarrow
\text{Factorized bimanual motion}
}
$$

其中：

- V1–V4 是 baseline / ablation；
- V5 是原始 event + phase 框架，也是 V6–V8 的历史基线；
- V6 将固定帧尺度改为 timestamp-derived physical duration，并锁定 state-only 输入契约；
- V7 将 phase crossing 边界修正为 successor trend onset；
- **V8 是当前推荐方法：Factorized Bimanual State-Only Atomic Motion Segmentation**；
- V1–V8 输出的核心仍是运动学原子运动段，不宣称完整 semantic action segmentation。

> 说明：本文 V1–V5 历史章节中出现的“最终 V5”，表示当时那轮迭代的结论；当前版本选择以文末 V6–V8 addendum 和 `V6_V8_ITERATION_new.md` 为准。

核心研究结论是：

> Pure geometric reconstruction is insufficient for atomic action segmentation. Instantaneous motion changes are also unreliable because robot trajectories are smoothly controlled. Robust segmentation therefore requires structured boundary evidence from manipulation events, supplemented by persistent motion-phase transitions when explicit pauses are absent.

---

# 第一部分：统一数学定义

## 1. 观测轨迹与状态空间

### 1.1 离散观测

设一个 episode 包含 $T$ 个带时间戳的离散观测：

$$
\mathcal X
=
\left\{
\left(t_i,\boldsymbol x_i\right)
\right\}_{i=0}^{T-1},
\qquad
 t_0<t_1<\cdots<t_{T-1}.
$$

这些观测是某条底层连续机器人轨迹的有噪离散采样。由于旋转属于 $SO(3)$，本文不把完整状态统一写成普通向量加性噪声模型。

如果需要分别形式化，可以写为：

$$
\boldsymbol p_i
=
\boldsymbol p^{\star}(t_i)
+
\boldsymbol\epsilon_i^p,
$$

以及：

$$
R_i
=
R^{\star}(t_i)
\operatorname{Exp}(\boldsymbol\xi_i),
\qquad
\boldsymbol\xi_i\in\mathfrak{so}(3).
$$

但当前算法并未显式估计噪声分布，因此后续只使用“有噪离散观测”这一较弱假设。

### 1.2 单臂和双臂状态

对机械臂 $h\in\{\mathrm L,\mathrm R\}$，定义物理状态：

$$
\boldsymbol x_i^h
=
\left(
\boldsymbol p_i^h,
R_i^h,
 g_i^h
\right)
\in
\mathbb R^3\times SO(3)\times[0,1].
$$

其中：

- $\boldsymbol p_i^h\in\mathbb R^3$：末端位置；
- $R_i^h\in SO(3)$：物理 orientation；
- $g_i^h\in[0,1]$：连续夹爪开合值。

完整双臂状态为：

$$
\boldsymbol x_i
=
\left(
\boldsymbol x_i^{\mathrm L},
\boldsymbol x_i^{\mathrm R}
\right)
\in
\left(
\mathbb R^3\times SO(3)\times[0,1]
\right)^2.
$$

实现中使用单位四元数 $\boldsymbol q_i^h\in S^3$ 表示 $R_i^h$，并显式处理双覆盖等价性：

$$
\boldsymbol q
\equiv
-\boldsymbol q.
$$

因此，$SO(3)$ 是数学状态空间，$S^3$ 四元数只是 numerical representation，二者不在同一个状态定义中混用。

---

## 2. Geometric knots 与 frame slicing boundary

### 2.1 几何折线节点

定义严格递增的 geometric knots：

$$
0
=
\kappa_0
<
\kappa_1
<
\cdots
<
\kappa_n
=
T-1.
$$

第 $k$ 个几何 primitive 是包含端点的区间：

$$
\mathcal G_k
=
[\kappa_{k-1},\kappa_k],
\qquad
k=1,\ldots,n.
$$

其近似轨迹由两个 knot state 连接：

$$
\boldsymbol x_{\kappa_{k-1}}
\longrightarrow
\boldsymbol x_{\kappa_k}.
$$

相邻 primitive 共享 knot：

$$
\mathcal G_k\cap\mathcal G_{k+1}
=
\{\kappa_k\}.
$$

因此，整个 approximation 是连续的 piecewise geodesic / polyline：

$$
\boldsymbol x_{\kappa_0}
\longrightarrow
\boldsymbol x_{\kappa_1}
\longrightarrow
\cdots
\longrightarrow
\boldsymbol x_{\kappa_n}.
$$

### 2.2 数据导出的帧归属约定

Geometric primitive 与数组 slicing 是两个不同概念。导出训练片段时，需要单独规定每一帧属于哪个 segment。

当前实现使用半开 ownership convention。若内部 slicing boundaries 为：

$$
0=b_0<b_1<\cdots<b_m=T,
$$

则导出片段为：

$$
[b_{j-1},b_j),
\qquad
j=1,\ldots,m.
$$

代码中的 inclusive frame 表示为：

$$
[b_{j-1},b_j-1].
$$

在 V1/V2 中，内部 geometric knot 可以被选作下一导出 slice 的起点，但这只是 frame ownership convention，不改变几何 primitive 共享 knot 的数学定义。

---

## 3. Piecewise-geodesic reconstruction cost

### 3.1 片段插值

考虑一个候选几何 primitive $[a,b]$，其中 $a<b$。定义：

$$
\alpha_i
=
\frac{t_i-t_a}{t_b-t_a},
\qquad
 i=a,\ldots,b.
$$

固定频率时，也可以使用：

$$
\alpha_i
=
\frac{i-a}{b-a}.
$$

位置线性插值：

$$
\widehat{\boldsymbol p}_i^h
=
(1-\alpha_i)\boldsymbol p_a^h
+
\alpha_i\boldsymbol p_b^h.
$$

旋转球面线性插值：

$$
\widehat R_i^h
=
\operatorname{SLERP}
\left(
R_a^h,
R_b^h,
\alpha_i
\right).
$$

实现中对相应四元数执行 SLERP，并先选择符号一致的表示。

夹爪线性插值：

$$
\widehat g_i^h
=
(1-\alpha_i)g_a^h
+
\alpha_i g_b^h.
$$

### 3.2 三类偏差

位置偏差：

$$
d_p
\left(
\boldsymbol p_i^h,
\widehat{\boldsymbol p}_i^h
\right)
=
\left\|
\boldsymbol p_i^h
-
\widehat{\boldsymbol p}_i^h
\right\|_2.
$$

旋转使用 $SO(3)$ 测地角。若用四元数表示：

$$
d_R
\left(
R_i^h,
\widehat R_i^h
\right)
=
2\arccos
\left(
\left|
\left\langle
\boldsymbol q_i^h,
\widehat{\boldsymbol q}_i^h
\right\rangle
\right|
\right).
$$

绝对值保证 $\boldsymbol q$ 与 $-\boldsymbol q$ 的旋转等价性。

夹爪偏差：

$$
d_g
\left(
 g_i^h,
\widehat g_i^h
\right)
=
\left|
 g_i^h-
\widehat g_i^h
\right|.
$$

### 3.3 Characteristic deviation scales

引入：

$$
s_p>0,
\qquad
s_R>0,
\qquad
s_g>0.
$$

它们不是没有物理含义的纯 normalization constants，而是 **characteristic deviation scales / tolerance scales**。

例如：

$$
s_p=0.01\ \mathrm m
$$

表示 1 cm 是一个具有实际意义的位置偏差尺度。当前默认值为：

$$
s_p=0.01\ \mathrm m,
\qquad
s_R=0.05\ \mathrm{rad},
\qquad
s_g=0.10.
$$

### 3.4 Robust reconstruction cost

单帧、单臂重构代价为：

$$
\ell_i^h(a,b)
=
w_p\rho
\left(
\frac{d_p}{s_p}
\right)
+
w_R\rho
\left(
\frac{d_R}{s_R}
\right)
+
w_g\rho
\left(
\frac{d_g}{s_g}
\right).
$$

Huber loss 定义为：

$$
\rho_\delta(z)
=
\begin{cases}
\dfrac12 z^2,
& |z|\leq\delta,
\\[4pt]
\delta\left(|z|-\dfrac12\delta\right),
& |z|>\delta.
\end{cases}
$$

双臂 primitive 的 reconstruction cost 为：

$$
C_{\mathrm{rec}}(a,b)
=
\sum_{h\in\{\mathrm L,\mathrm R\}}
\sum_{i=a}^{b}
\ell_i^h(a,b).
$$

---

## 4. Within-segment motion dispersion

定义局部运动特征：

$$
\boldsymbol f_i^h
=
\begin{bmatrix}
\boldsymbol v_i^h/s_v
\\
\boldsymbol\omega_i^h/s_\omega
\\
\dot g_i^h/s_{\dot g}
\end{bmatrix},
$$

其中：

$$
\boldsymbol v_i^h
=
\frac{
\boldsymbol p_{i+1}^h-
\boldsymbol p_i^h
}{t_{i+1}-t_i},
$$

$$
\boldsymbol\omega_i^h
=
\frac{
\operatorname{Log}
\left(
(R_i^h)^\top R_{i+1}^h
\right)
}{t_{i+1}-t_i},
$$

$$
\dot g_i^h
=
\frac{g_{i+1}^h-g_i^h}{t_{i+1}-t_i}.
$$

可定义 within-segment motion dispersion：

$$
C_{\mathrm{disp}}(a,b)
=
\sum_{h}
\sum_{i=a}^{b-1}
\left\|
\boldsymbol f_i^h-
\overline{\boldsymbol f}_{a:b}^h
\right\|_2^2.
$$

该代价表达的是 segment 内平移速度、角速度和夹爪速度是否接近一个稳定 regime，也可以称为 constant-motion consistency cost。

但必须明确：

$$
\text{low motion dispersion}
\not\Rightarrow
\text{semantic or kinematic atomicity}.
$$

真实 atomic motion 仍可能包含 accelerate–cruise–decelerate。因此，motion dispersion 可以作为辅助 inductive bias，不能单独定义 atomic action。

---

## 5. 概念优化问题与结构化实现

### 5.1 多目标冲突

几何层面的基本目标是同时减小 primitive 数量和重构误差：

$$
\min
\left(
 n,
\sum_{k=1}^{n}
C_{\mathrm{rec}}
(\kappa_{k-1},\kappa_k)
\right).
$$

标量化形式可以写为：

$$
\min_{n,\boldsymbol\kappa}
\quad
\sum_{k=1}^{n}
\left[
C_{\mathrm{rec}}
(\kappa_{k-1},\kappa_k)
+
\beta C_{\mathrm{disp}}
(\kappa_{k-1},\kappa_k)
\right]
+
\lambda n.
$$

它适合描述 geometric complexity，但不足以完整表达 action-transition evidence。

### 5.2 结构化 boundary-evidence formulation

对 slicing boundary $b$，定义：

- $z_{\mathrm{pause}}(b)$：持续低运动能量或 valley 证据；
- $z_{\mathrm{gripper}}(b)$：夹爪事件及其稳定区证据；
- $z_{\mathrm{phase}}(b)$：持续 motion-phase transition 证据。

可写出概念性目标：

$$
\min_{\mathcal B}
\quad
E_{\mathrm{rec}}(\mathcal B)
+
\lambda|\mathcal B|
-
\eta_p
\sum_{b\in\mathcal B}
z_{\mathrm{pause}}(b)
-
\eta_g
\sum_{b\in\mathcal B}
z_{\mathrm{gripper}}(b)
-
\eta_s
\sum_{b\in\mathcal B}
z_{\mathrm{phase}}(b).
$$

当前实现没有直接联合优化所有连续 trade-off weights。更准确的说法是：

> Rather than directly optimizing all continuous trade-off weights, the implemented procedure realizes the same modeling principle through structured candidate generation and evidence-based filtering.

或者：

> The implemented procedure can be interpreted as a structured discrete approximation to the conceptual objective above.

它不是该目标的 mathematically equivalent solver。

### 5.3 约束

边界必须满足完整覆盖和时间顺序：

$$
0=b_0<b_1<\cdots<b_m=T.
$$

最短导出片段约束：

$$
b_j-b_{j-1}
\geq
L_{\min}.
$$

相邻候选的 temporal merge 半径为 $r_{\mathrm{merge}}$。Soft phase 必须满足 persistence：

$$
\operatorname{duration}(\phi_{\mathrm{before}})
\geq L_{\mathrm{phase}},
$$

$$
\operatorname{duration}(\phi_{\mathrm{after}})
\geq L_{\mathrm{phase}}.
$$

对于本数据：

$$
\Delta t\approx0.02\ \mathrm s,
\qquad
f_s\approx50\ \mathrm{Hz}.
$$

低于采样和滤波时间尺度的动作变化无法被稳定恢复。

---

# 第二部分：Atomicity 层级

## 6. 三层 atomicity

### 6.1 Level 1：Geometric primitive

轨迹的一段可以由简单 piecewise geodesic 描述。

V1/V2 主要对应这一层，其问题是：

> How many simple geometric segments are required to reconstruct the trajectory?

### 6.2 Level 2：Kinematic atomic primitive

片段内部具有相对稳定、可解释的运动模式，例如：

- move；
- turn；
- lift；
- lower；
- grasp / open gripper。

V4/V5 主要解决这一层。

### 6.3 Level 3：Semantic action

包含对象、接触状态和任务语义，例如：

- approach bottle；
- grasp bottle；
- lift bottle；
- hammer nail；
- place cup。

只使用末端位姿、夹爪和 timestamp，无法可靠恢复 Level 3。完整语义动作还需要：

- task instruction；
- image / video；
- object tracking；
- contact / force；
- attachment state；
- learned visual representation。

因此最终方法定位为：

$$
\boxed{
\text{kinematic atomic motion segmentation}
}
$$

而不是 full semantic action segmentation。

---

# 第三部分：V1–V5 方法演进

## 7. 方法角色总览

$$
\boxed{
\underbrace{\text{V1/V2}}_{\text{trajectory approximation}}
\rightarrow
\underbrace{\text{V3}}_{\text{motion change}}
\rightarrow
\underbrace{\text{V4}}_{\text{strong event evidence}}
\rightarrow
\underbrace{\mathbf{V5}}_{\mathbf{strong\ events+persistent\ phases}}
}
$$

这五个版本不是五个平级最终候选，而是一条 inductive-bias 演进链。

---

## 8. V1：Piecewise-geodesic trajectory approximation baseline

固定 primitive 数量 $K$，动态规划求：

$$
E_K
=
\min_{
0=\kappa_0<\cdots<\kappa_K=T-1
}
\sum_{k=1}^{K}
C_{\mathrm{rec}}
(\kappa_{k-1},\kappa_k).
$$

固定 $K$ 时，DP 对 reconstruction objective 给出全局最优 geometric knots。

V1 回答的是：

> How many simple geometric primitives are required to reconstruct the trajectory?

但：

$$
\boxed{
\text{geometric complexity}
\neq
\text{action complexity}
}
$$

平滑控制轨迹中的加速、弯曲和减速可能导致几何复杂度升高，但不一定是动作切换；反过来，不同语义或运动学阶段也可能被 controller 平滑连接。

因此 V1 是 mathematical baseline，不是最终 atomic motion segmentation。

---

## 9. V2：Geometry-aware reconstruction baseline

V2 在 V1 基础上修正：

- $SO(3)$ geometry；
- quaternion sign equivalence；
- rotation SLERP；
- continuous gripper；
- characteristic deviation scales；
- Huber robust loss。

V2 的作用不是成为最终方法，而是验证：V1 的失败并不只是 representation 错误。

即使状态空间和损失被正确处理：

$$
E_{\mathrm{rec}}+\lambda n
$$

仍无法可靠恢复动作边界。纯 reconstruction objective 缺少 manipulation event 和 phase persistence 的 inductive bias。

---

## 10. V3：Motion change-point ablation

V3 从 trajectory domain 转向 derivative domain：

$$
\boldsymbol f_t
=
\begin{bmatrix}
\boldsymbol v_t
\\
\boldsymbol\omega_t
\\
\dot g_t
\end{bmatrix}.
$$

变化分数例如：

$$
c_t
=
\left\|
\boldsymbol f_t-
\boldsymbol f_{t-1}
\right\|_2.
$$

阈值使用 median 和 MAD：

$$
\theta_c
=
\operatorname{median}(c)
+
\gamma
\operatorname{MAD}(c).
$$

V3 的失败原因是：

$$
\boxed{
\text{instantaneous motion change}
\neq
\text{action boundary}
}
$$

正常 acceleration / deceleration 会产生大量变化点，而真实 action transition 又可能被 controller 平滑。V3 因此保留为 dynamics ablation。

---

## 11. V4：Strong event-evidence ablation

V4 不再寻找任意 numerical change point，而显式寻找 manipulation 中具有意义的 strong boundary evidence。

对每只手臂，定义平滑运动能量：

$$
e_t^h
=
\frac{
\|\Delta\boldsymbol p_t^h\|_2
}{s_{\Delta p}}
+
\frac{
d_R(R_t^h,R_{t-1}^h)
}{s_{\Delta R}}
+
\frac{
|\Delta g_t^h|
}{s_{\Delta g}},
$$

$$
\widetilde e_t^h
=
\operatorname{Smooth}(e_t^h).
$$

Strong boundary evidence 集合定义为：

$$
\mathcal B_{\mathrm{strong}}
=
\mathcal B_{\mathrm{pause}}
\cup
\mathcal B_{\mathrm{gripper}}.
$$

它包括：

1. sustained low-motion pause；
2. local motion-energy valley；
3. gripper close/open event；
4. gripper event 周围的 pre/post stabilization valley。

V4 的核心思想是：

$$
\boxed{
\text{action-transition evidence}
>
\text{generic trajectory change}
}
$$

V4 只使用 strong evidence，因此保留为 event-only ablation。

---

## 12. V5：Event- and Phase-Aware Atomic Motion Segmentation

### 12.1 最终关系

V5 不再与 V4 并列选择，而被定义为 V4 的 superset：

$$
\boxed{
\text{V5}
=
\text{V4 strong event backbone}
+
\text{persistent motion-phase evidence}
}
$$

### 12.2 Strong evidence

$$
\mathcal B_{\mathrm{strong}}
=
\mathcal B_{\mathrm{pause}}
\cup
\mathcal B_{\mathrm{gripper}}.
$$

Strong candidate 保存：

- source arm；
- evidence type；
- evidence class；
- strength；
- related gripper interval；
- 是否被双臂共同支持。

### 12.3 Windowed motion phase

Motion phase 集合为：

$$
\Phi
=
\{
\texttt{still},
\texttt{move},
\texttt{turn},
\texttt{lift},
\texttt{lower},
\texttt{close\_gripper},
\texttt{open\_gripper}
\}.
$$

Phase 不由单帧 displacement 分类。对局部窗口：

$$
[t-r,t+r],
$$

计算：

- net displacement；
- path length；
- vertical displacement；
- horizontal displacement；
- vertical / horizontal ratio；
- accumulated rotation；
- translation-vs-rotation dominance；
- gripper trend。

定义单位 vertical direction：

$$
\widehat{\boldsymbol v}_{\mathrm{vertical}}
\in\mathbb R^3,
\qquad
\|\widehat{\boldsymbol v}_{\mathrm{vertical}}\|_2=1.
$$

窗口净位移为 $\Delta\boldsymbol p_t$，vertical component 为：

$$
\Delta p_{\mathrm{vertical},t}
=
\Delta\boldsymbol p_t^\top
\widehat{\boldsymbol v}_{\mathrm{vertical}}.
$$

horizontal component 为：

$$
\Delta\boldsymbol p_{\mathrm{horizontal},t}
=
\Delta\boldsymbol p_t
-
\Delta p_{\mathrm{vertical},t}
\widehat{\boldsymbol v}_{\mathrm{vertical}}.
$$

当前 RoboTwin world $z$ 与 gravity 对齐，因此：

$$
\widehat{\boldsymbol v}_{\mathrm{vertical}}
=
(0,0,1)^\top.
$$

迁移到其他坐标系时应显式配置该方向，而不是永久依赖 world $z$。

### 12.4 Phase persistence

不能因为：

$$
\phi_t\neq\phi_{t-1}
$$

就直接切分。需要检测类似：

```text
A A A A A  ->  B B B B B B
```

并过滤：

```text
A A A B A A A
```

形式化地，soft candidate $b$ 至少需要：

$$
\operatorname{duration}
(\phi_{\mathrm{before}})
\geq
L_{\mathrm{phase}},
$$

$$
\operatorname{duration}
(\phi_{\mathrm{after}})
\geq
L_{\mathrm{phase}},
$$

并且前后 phase 差异具有运动意义，且不是短时 oscillation。当前默认实现只在该臂最后一个显式 gripper event 之后启用 soft phase 候选，以避免 episode 开头的平滑接近轨迹被过度细分；这是一个保守的可配置先验，而不是 phase 定义本身。

Soft evidence 定义为：

$$
\mathcal B_{\mathrm{soft}}
=
\mathcal B_{\mathrm{phase}}.
$$

典型 transition 包括：

$$
\texttt{lift}\rightarrow\texttt{move},
$$

$$
\texttt{move}\rightarrow\texttt{lower},
$$

$$
\texttt{move}\rightarrow\texttt{turn}.
$$

### 12.5 Evidence fusion 与最终段数

不能把 pause、gripper 和 phase candidate 数量直接相加，因为同一个 boundary 可能同时拥有多种 evidence。

最终定义为：

$$
\boxed{
\mathcal B
=
\operatorname{FilterMerge}
\left(
\mathcal B_{\mathrm{strong}}
\cup
\mathcal B_{\mathrm{soft}}
\right)
}
$$

其中 `FilterMerge` 包括：

1. minimum segment duration；
2. temporal non-maximum suppression / nearby-boundary merge；
3. phase persistence；
4. strong-over-soft priority；
5. isolated noisy phase-transition removal；
6. 多 evidence provenance fusion；
7. 双臂 candidate temporal merge。

最终：

$$
\boxed{
n
=
1+|\mathcal B|
}
$$

这才是 V5 的最终自动段数定义。

---

# 第四部分：双臂 formulation

## 13. 从 global active arm 到 joint evidence fusion

旧简化方法选择：

$$
h^\star
=
\underset{h}{\operatorname{argmax}}
\sum_t e_t^h.
$$

它适合单主操作臂任务，但不能作为完整双臂 formulation。

最终 V5 对左右臂分别得到：

$$
\mathcal B_{\mathrm L},
\qquad
\mathcal B_{\mathrm R}.
$$

再执行：

$$
\mathcal B_{\mathrm{joint}}
=
\operatorname{TemporalMerge}
\left(
\mathcal B_{\mathrm L}
\cup
\mathcal B_{\mathrm R}
\right).
$$

每个最终 boundary 保留：

- `source_arms`；
- `evidence_types`；
- `evidence_class`；
- `strength`；
- `supported_by_both_arms`；
- 原始 support candidate 列表。

当前代码默认：

```text
--arm-mode joint
```

旧 active-arm 方法保留为：

```text
--arm-mode active
```

仅用于简化模式或 ablation。

---

# 第五部分：数据与实验解释

## 14. 数据事实

用户提供的 root：

```text
/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0
```

实际是 14D 双臂关节空间：每臂 6 个关节值和连续夹爪。

严格配对的 end-pose 数据位于：

```text
/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/cosmos3/data/robotwin2.0-endpose
```

每臂为：

```text
xyz + quaternion(wxyz) + continuous gripper
```

时间戳真实存在，采样频率约 50 Hz。

---

## 15. Weak reference

现有参考文件：

```text
/mnt/ybw/workspace/divide_action/outputs/full/atomic_segments_cot_merged_grasp_task01.json
```

它来自已有自动规则与语义合并，不是人工逐帧 Ground Truth。因此必须称为：

> weak segmentation reference

当前 boundary F1 只能解释为：

> agreement with the existing weak segmentation reference.

不能解释为真实 segmentation accuracy。

---

## 16. 现有两条样例的消融现象

### 16.1 Bottle episode 0

已有 weak reference：

```text
turn -> move -> grasp -> lift -> turn
```

V1/V2 表明 geometric reconstruction 会合并动作阶段或按轨迹曲率切分。V4 能找到主要 pause / gripper / lift 边界。V5 在 V4 基础上补充 persistent phase evidence，但也可能将末端复合转动中的稳定 vertical correction 切成更细的 kinematic primitive。

这说明 V5 的目标不是最大化 weak-reference F1，而是形成统一、可审计的强/软证据框架。是否保留更细 phase，最终需要人工标注和 downstream evaluation 决定。

### 16.2 Hammer episode 550

V5 输出：

```text
move -> grasp -> lift -> move -> lower
```

在该样例中，strong valley 已足以找到主要边界；persistent phase evidence 与 strong evidence 在 `lift -> move`、`move -> lower` 附近重合，并被融合为同一个 boundary，而不会重复计数。

### 16.3 小规模 weak-reference agreement

历史只读检查覆盖：

- bottle episode 0–19；
- hammer episode 550–569。

这些结果只能作为方法调试信号，不能替代人工 benchmark。特别禁止为了提高两类 episode 的 weak-reference F1 而对规则进行任务特化。

---

# 第六部分：评价体系

## 17. Intrinsic evaluation

下一轮人工标注应优先只标 boundary，每个主要 task 建议 10–20 个 episode。

至少报告：

- boundary precision；
- boundary recall；
- boundary F1；
- tolerance-based boundary F1；
- mean / max boundary localization error；
- over-segmentation count；
- under-segmentation count；
- signed segment count error；
- absolute segment count error。

当前代码已经针对 weak reference 输出上述可计算指标，并在字段中明确 `reference_type` 和解释说明。

粗粒度 action label 可以作为第二阶段标注和评价。

---

## 18. Extrinsic evaluation

最终必须验证 segmentation 是否为具身学习生成了更有价值的训练单元。

建议比较：

- fixed-length chunking；
- V1/V2 geometric primitives；
- V4 strong-event only；
- V5 strong events + persistent phases。

下游任务至少包括：

- action prediction；
- world-model prediction；
- temporal consistency；
- rollout success；
- policy performance。

最终判断标准不是单纯：

> segmentation 看起来是否像人。

而是：

$$
\boxed{
\text{Does the segmentation produce more useful training units for downstream embodied learning?}
}
$$

---

# 第七部分：统一最终流程

## 19. V5 pipeline

$$
\boxed{
\text{Raw end-effector trajectories}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Kinematic feature extraction}
}
$$

包括：

- translation；
- orientation / accumulated rotation；
- gripper；
- velocity / angular velocity；
- smoothed motion energy；
- local window displacement and path statistics。

然后分别生成：

$$
\boxed{
\text{Strong evidence}
=
\text{pause / valley}
+
\text{gripper stabilization}
}
$$

以及：

$$
\boxed{
\text{Soft evidence}
=
\text{persistent motion-phase transitions}
}
$$

之后：

$$
\boxed{
\text{Left/right evidence fusion}
}
$$

$$
\downarrow
$$

$$
\boxed{
\text{Temporal merge + persistence + minimum duration}
}
$$

$$
\downarrow
$$

$$
\boxed{
\mathcal B
=
\{b_1,\ldots,b_{n-1}\}
}
$$

$$
\downarrow
$$

$$
\boxed{
n=1+|\mathcal B|
}
$$

最终得到：

$$
\boxed{
\text{Kinematic atomic motion segments}
}
$$

---

# 第八部分：代码对应关系

## 20. 主要模块

```text
atomic_episode_segmentation/
├── atomic_seg/
│   ├── geometry.py       # quaternion / SO(3) numerical utilities
│   ├── features.py       # dual-arm features and local-window statistics
│   ├── segmentation.py   # V1–V5 and shared interval/evidence helpers
│   ├── state_segmentation.py # V6–V8 time/onset/bimanual methods
│   ├── evaluation.py     # weak-reference agreement metrics
│   ├── plotting.py       # dual-arm evidence/onset-aware SVG diagnostics
│   ├── pipeline.py       # unified version dispatch
│   └── io.py
├── tests/test_core.py
├── run_experiments.py
├── README.md
├── V6_V8_ITERATION_new.md
└── ITERATION_REPORT.md
```

V1/V2 的输出显式保存：

- `geometric_knots`；
- `geometric_primitives`；
- `frame_ownership_convention`。

V4/V5 的输出显式保存：

- `strong_candidates`；
- `soft_candidates`；
- `boundary_evidence`；
- evidence provenance 和双臂支持。

V6–V8 进一步保存：

- physical-time temporal parameters；
- state-only decision input policy；
- successor onset refinement；
- per-arm boundaries / primitive timelines；
- bimanual coordination relation。

---

# 第九部分：限制与下一步

## 21. Remaining limitations

1. Phase classifier 仍是可解释的 hand-designed kinematic rule，不是学习式语义模型；
2. Characteristic rate scales、pause threshold、phase persistence 等仍需人工标注集标定；
3. V8 已建模 per-arm timeline 与 coordination relation，但 state-only 仍不能确认 handover 的对象流和 donor/recipient；
4. 没有视觉、物体、接触和力信息，无法可靠恢复 Level 3 semantic action；
5. V1/V2 的自动 $K$ 选择仍是 baseline elbow heuristic，不应被视为唯一最优模型选择准则；
6. Weak-reference F1 只表示 agreement，不表示真实准确率；
7. 最终价值必须由 downstream embodied-learning performance 验证。

---

## 22. 复现实验

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation

python -m unittest discover -s tests -v

python run_experiments.py \
  --data-root /apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0 \
  --episodes 0,550,6100 \
  --versions 1,2,3,4,5,6,7,8 \
  --arm-mode joint \
  --vertical-direction 0,0,1 \
  --output-dir outputs
```

---

# 第十部分：V6–V8 Addendum

## 23. V6：Time-Normalized State Evidence

V6 保留 V5 的 strong event + persistent phase 思路，但做三项基础修正：

1. 所有持续时间从 timestamp 换算，不再假设固定 50 Hz；
2. 速度使用 m/s、rad/s、normalized gripper amplitude/s；
3. 切分输入契约明确为 state + timestamp，视觉和 `action_config` 不参与决策。

V6 的边界函数为：

$$
\mathcal B_6
=
F_6
\left(
\left\{
t_i,
\boldsymbol p_i^L,R_i^L,g_i^L,
\boldsymbol p_i^R,R_i^R,g_i^R
\right\}_{i=0}^{T-1}
\right).
$$

不包含：

$$
\mathrm{video},
\quad
\mathrm{action\_config},
\quad
\mathrm{text}.
$$

gripper 只按 robust normalized amplitude 检测 increase / decrease，不在 motion boundary 层猜测 open / close 极性。

## 24. V7：Successor-Trend-Onset

V6 的 persistent phase transition 仍由 centered local window 产生，边界容易接近 phase score crossing。

若旧动作分数下降、新动作分数上升：

$$
\dot s_{\mathrm{old}}(t)<0,
\qquad
\dot s_{\mathrm{new}}(t)>0,
$$

V7 定义：

$$
b
=
\min
\left\{
t:
s_{\mathrm{new}}(t)
\text{ 已出现持续、可确认的上升支持}
\right\}.
$$

即：

$$
\boxed{
\text{boundary = successor onset}
}
$$

而不是：

$$
\text{boundary = score crossing}
$$

或：

$$
\text{boundary = predecessor completion}.
$$

输出保存 nominal frame、refined frame、前后 score 和 competing-trend diagnostics，便于人工检查。

## 25. V8：Factorized Bimanual Motion

V8 首先独立产生：

$$
\mathcal B^L,\mathcal P^L
$$

和：

$$
\mathcal B^R,\mathcal P^R.
$$

然后派生 coordination timeline：

$$
\mathcal B^C
=
G
\left(
\mathcal B^L,
\mathcal B^R
\right).
$$

其中：

- per-arm timeline 是 primary result；
- coordination timeline 是 derived result；
- 同步且兼容的 evidence 可以共享边界；
- sequential 或不同 phase 的 evidence 尽量保持分离；
- 即使 joint view 因 minimum duration 合并近邻边界，per-arm primary evidence 仍保留。

V8 joint segment relation：

```text
both_still
left_only
right_only
dual_same_phase
dual_different_phase
```

## 26. V6–V8 与语义标注的边界

推荐生产链：

```text
V8 state segmentation
→ freeze state boundaries
→ optional video/data-config annotation
→ caption/instruction/object/target/semantic skill
```

若语义层希望提出不同的 operation window，应另存：

```text
semantic_window_proposal
```

不能静默覆盖：

```text
state_motion_boundary
```

## 27. V6–V8 可视化

新版 SVG 同时绘制：

- left/right xyz；
- left/right gripper；
- left/right state energy；
- left/right phase lane；
- joint segment labels；
- strong / phase / onset boundary；
- weak reference。

标记：

```text
S = strong state event
P = persistent phase
O = successor onset
```

完整方法说明、JSON 字段和视觉检查流程见：

```text
V6_V8_ITERATION_new.md
```
