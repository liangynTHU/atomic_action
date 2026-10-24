# OXE / ABC-130k 预训练非 CoT Pipeline

RealOmni Open 使用独立的整集检查与可恢复分割入口 `scripts/prepare_realomni.py`，说明见 [REALOMNI_PREPARATION.md](REALOMNI_PREPARATION.md)。其 `realomni_umi` 合同为相邻实测末端状态派生的 16D 参考 action／guide；底盘筛选按用户澄清取消，缺失样本整条排除。

入口：`python scripts/run_pipeline.py --config configs/pretrain_noncot.same_actions.json`。
独立入口 `scripts/generate_pretrain_noncot.py` 支持 `prepare / annotate / export / review / all` 分步执行。

本流程复用 `atomic_seg.segmentation.segment_episode`、通用关节运动候选函数、公共模型客户端、JSON 写入及 writer lock。可视化复用 `scripts/episode_dataset_review.html` 的样式、字体与完整 episode 审阅设计。旧 ABC V29 专项流程不参与。

## 当前合并规则：只合并相邻同动作片段

2026-10-21 更新 ABC、AgiBot、RealOmni 共用的 `pretrain/length_policy.py`：相邻片段的 `left_action` 和 `right_action` 必须分别相同才允许合并。`close_gripper → close_gripper` 可合并，`close_gripper → lift`、`move → turn` 以及任一侧的动作标签变化都保留边界。32–64 行只是同动作内部的长度偏好，不能用来跨动作凑长度；显式硬下限与动作边界冲突时仍报错。

既有双臂 still 吸收规则继续生效，静止上下文以 `absorbed_still_spans` 保留来源；它不允许把前后不同的活动动作合为一段。合并后的标签直接保留共同动作，不再从多个阶段中优先选取夹爪动词。`length_audit.merge_policy=identical_bimanual_actions_v1`，新增 `action_boundaries`；不足偏好长度的片段记录 `different_action_labels` 等保留原因。

推荐配置 `configs/pretrain_noncot.same_actions.json` 使用新的输出目录，可复用已解码源数据，但重新分段、标注并按新边界生成 guide、chunks、历史和图片。标注与导出入口拒绝带有混合动作 primitive 的旧结果。AgiBot/RealOmni 的代码指纹也会变化，重跑须使用新 run 目录。既有发布数据和审阅 HTML 不会自动更新，下面的 2026-10-13 统计属于旧合并策略。

## 历史版本：2026-10-13 静止段统一处理，长度采用软偏好

2026-10-13 按用户澄清更新。当时使用配置 `configs/pretrain_noncot.soft_chunks_still.json`，输出 `artifacts/pretrain_noncot_20261013_soft_chunks_still`。该配置和旧 16 帧硬下限配置保留作历史记录；复现历史分段需要对应旧代码。

- `minimum_frames=null`，默认无 16 帧硬下限；仍偏向 32–64 行的片段，合并相邻微小动作。短动作因任务边界、相反夹爪动作或终止 hold 有独立意义时可保留，并记录原因。不会直接抬高 num_chunks 或复制帧。
- ABC 位姿分支和关节分支均经过 `abc_still.absorb_both_still`，复用 `plan_still_segment_groups`。初始双臂 still 合入下一动作，中间 still 合入上一动作，末尾不足 8 行时吸收，否则允许一个 terminal hold；整个 episode 都静止时保留一个 hold。所有原始行保留，来源和吸收角色可追溯。
- 长度合并保留 terminal hold 边界，避免仅为扩大 chunks 把 hold 吞掉。原始 subtask 标注仍映射到最近的既有边界；这只是运动/标注约束，不声称做了完整语义审核。
- 关节分支判定 still 时同时检查关节路径和夹爪范围，开闭爪后回到原值也不会因为净 delta=0 被误判完全静止。
- ABC 学生和 teacher 的 14D 顺序明确为左 joint1–6、左 gripper、右 joint1–6、右 gripper。关节角用 radians、base-to-wrist 顺序；gripper 是绝对 aperture，0=closed、1=open；guide 是最后关节命令减当前状态，不是末端 xyz。不能从关节 delta 正负推断 Cartesian 运动方向。

4 条样本重算后最终仍为 86 段，边界和主要动作标签恰好与上一版一致，因此复用旧子任务文本；只保留文本，吸收静止段和 primitive 来源使用当前新结果。杯子示例的开头 0–18（19 行）及中间 1572–1592（21 行）共 40 行已进入显式 still 吸收记录。有位姿样本更早在核心算法中吸收的 still 另存在 `core_still_refinement`。

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/run_pipeline.py --config configs/pretrain_noncot.soft_chunks_still.json
```

## 对齐 RoboTwin 的学生训练格式

使用 `configs/pretrain_noncot.robotwin_aligned.json`，输出为 `artifacts/pretrain_noncot_20261012_robotwin_aligned`。本次保留上一版 4 条 OXE、4 条 ABC 的分段与目标，复用真实 Qwen 子任务标注，仅重建学生 system/user、训练行格式和可视化。

OXE 与 ABC 共用 `pretrain_noncot` Pipeline。数据读取分别由 RLDS/TFRecord 和 MCAP adapter 完成，原始坐标与分段适配按 profile 选择；其后的标注、学生消息构建、导出、校验、可视化共用代码。增加其他 OXE 子集需要明确字段适配，不能只替换路径。

- 训练 JSONL 每行只有 `messages` 和 `images`，与 RoboTwin 一致；id、dataset、split、帧号、图片角色、原始状态等放在 `metadata/{preview,train,val}.jsonl`，与训练行一一对应。
- 学生 system 是一个完整段落，共同 planner 句子来自 `atomic_seg/prompt_contract.py`：下一原子动作、guide 起终点和残差重算、8-action chunk、手臂约束、比较历史执行与观测变化、重复/恢复/推进、四字段 JSON。RoboTwin 既有 CoT system 文本哈希保持不变。
- 非 CoT system 移除 thinking 段落指令，不让学生输出解释正文。OXE 保留真实单臂 7D xyz/Euler rpy/gripper，ABC 保留双臂 14D joints/gripper；不能把它们写成 RoboTwin 的 16D xyz/quaternion。
- `user` 使用 RoboTwin 顺序：前缀图片标记、Task、Images、首帧无历史说明或 Previous command / Previous execution / Previous boundary state、Current state、Observed state delta，最后复用 `NO_THINKING_USER_ENDING`。ABC 显式允许多臂动作。
- 当前/历史状态按臂命名，显示四位小数，与 RoboTwin 展示方式一致；原始数值在 metadata/VLA/trajectory 中保留。Previous execution 的有效行数由相邻决策边界之差计算，padding 不计入有效行。
- `assistant` 直接输出含空格的四字段 JSON，没有 `<think>`。原 `sub_task`、guide 数值和 chunks 保持不变；RoboTwin 的动作标签本身使用冒号，这些标签格式不改。
- 训练 `images` 是存在的绝对路径。网页加载 metadata 中随包相对图片路径，离线包可以移动后打开。训练文件若迁移到其他机器仍需要按实际位置映射路径。

`reuse_annotations_root` 显式指定旧标注目录。复用时核对源文件、状态、动作、任务、源标注及分段一致性，并保留原始模型请求；它不重新调用模型，也不将旧学生 prompt 当作新训练数据。复现命令：

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/run_pipeline.py --config configs/pretrain_noncot.robotwin_aligned.json
```

页面中有当前完整 system 和 RoboTwin 原文对照、仅含 messages/images 的训练样本，以及独立 metadata。格式报告存 `STUDENT_FORMAT.json`。

## 历史版本：ABC 最少 16 帧，优先 32 帧及以上

配置为 `configs/pretrain_noncot.min16_pref32.json`，输出到独立目录 `artifacts/pretrain_noncot_20261012_min16_pref32`。

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/run_pipeline.py --config configs/pretrain_noncot.min16_pref32.json
```

ABC 的 `length_policy` 指定 `minimum_frames=16`、`preferred_minimum_frames=32`、`preferred_maximum_frames=64`、`maximum_merged_frames=128`。16 是有效动作行硬下限，32 是软目标，32–64 为偏好范围；自然边界会产生更长片段。仅移除旧 motion primitive 之间的边界，不插值、不复制动作、不改变 chunk size。全局代价同时考虑短段、过长段、运动变化；同一只手的 open/close 不能合并。原有 subtask 标注保留最近的既有 motion boundary，并记录吸附偏移，不能宣称与原始标注帧完全相同。遇到硬下限与保护边界冲突时明确报错，留给复核。

每段保存 `primitive_segments`、按臂 `motion_sequence`、`boundary_context`；16–31 帧片段另存 `short_segment_reason`。Qwen 接收 5 张时序拼图和完整 primitive 顺序，为合并后的整个局部动作重新生成标签。`guide_action`、`num_chunks`、上一决策边界和输入图片均按新边界重建。

`reuse_prepared_root` 只复用已解码的同一源 episode，重新核对 MCAP/TFRecord 哈希；OXE 无长度策略时核对所有边界和标签后复用原标注。旧输出保持可对照。新 index 默认打开 ABC 毛巾任务，包含新旧 chunks 柱图、每 episode 帧数表、短段标记和合并详情，统计另存 `LENGTH_COMPARISON.json`。

本轮 ABC 4 条共 4,947 个有效动作行：221 → 86 段，81/86（94.2%）达到 32 帧以上，最短 17 帧。木块 36 → 9、毛巾 119 → 37、白板 10 → 6、杯子 56 → 34。OXE 4 条保持 22 段。

## 输入及输出合同

| 数据 | 实际输入 | 学生 state / guide | native action | 特别处理 |
|---|---|---|---|---|
| OXE / Bridge | RLDS TFRecord，每个 Example 一个 episode | 7D xyz / Euler rpy / aperture | 平移增量、旋转增量、绝对 open_gripper | 一个外部相机；RLDS 没有时间戳，显式声明 nominal_fps=5；TFRecord CRC 验证 |
| ABC / YAM | episode.mcap，可选 annotation.mcap | 14D，左右各 6 joint + gripper | 绝对 joint / aperture 命令 | 原始独立时钟；状态和图像取 action 时刻及之前最近观测；保留全部 action rows |

OXE 是集合，本轮只实现和测试 Bridge profile；其他 OXE 子集需要增加自己的字段映射与 prompt，不能静默套 Bridge。ABC 读取实际 video codec，支持 H264/H265 以及 mono/stereo 相机 topic。源 MCAP 文档说 RobotState 可含 pose，但本地部分文件的 pose 为空：有完整 pose 的样本复用端位姿分段；没有 pose 的样本复用关节运动候选和实测夹爪变化，不伪造 Cartesian 位姿。

分段结果覆盖完整 native action 行；起止下标均包含端点。`num_chunks=ceil(rows/8)`。动作原值原维度保存，不实际写入 padding；`valid_action_rows` 与 `padding_rows_if_chunked` 供下游正确加 mask。

- ABC：`guide_action = action[end] - state[start]`，14D。
- Bridge：先计算末行命令目标 `[state[end,:6]+action[end,:6], open_gripper[end]]`，再减 `state[start]`，7D。这里是命令目标差，不把 delta action 当绝对位姿。

## Qwen 和 prompt

`atomic_seg/pretrain/prompts.py` 按数据集分别定义 state / action / camera / gripper 语义。学生 `system` 同时声明 guide 的含义与维度，`assistant` 严格只有 `guide_action, primary_action_verb, sub_task, num_chunks`，直接 JSON，无 CoT。

Qwen 仅给固定分段生成 `sub_task`。请求显式 `enable_thinking=false`，拒绝隐藏 reasoning、额外 key、不匹配的 segment index、思考标签或解释性输出。它不能写分段边界、guide 或 chunks。请求默认每批 4 段、温度 0.1；失败最多重试 3 次。真实请求文本、参数、图片 hash 和原始响应存 `requests/`，成功结果按输入 hash 缓存。

Teacher 看片段起中末的多相机拼图；学生仅看当前及上一决策边界的图像与状态。Teacher 的未来证据只用于离线动作命名，不进入学生 user。原始 subtask annotation 只用作 coarse context，完整保存用于查看。

## 运行

```bash
python -m pip install -e '.[pretrain]'
python scripts/generate_pretrain_noncot.py --config configs/pretrain_noncot.pilot.json --dry-run
python scripts/generate_pretrain_noncot.py --config configs/pretrain_noncot.pilot.json --stage prepare
python scripts/generate_pretrain_noncot.py --config configs/pretrain_noncot.pilot.json --stage annotate
python scripts/generate_pretrain_noncot.py --config configs/pretrain_noncot.pilot.json --stage export
python scripts/generate_pretrain_noncot.py --config configs/pretrain_noncot.pilot.json --stage review
```

本环境实际使用 `/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python`，新增依赖隔离在 `.runtime/pretrain_deps`，运行前设置 `PYTHONPATH=.runtime/pretrain_deps`。将 endpoint_file 指向可用 Qwen 服务列表。示例中 `.runtime/pretrain_endpoints.txt` 是本次运行现场配置，不属于可迁移的默认服务地址。端点只复用，不由这个脚本部署或停止。

输出目录及来源配置保持不变时可以重复运行，已解码 episode、成功标注和完整视频缓存可复用。改变数据映射、prompt、模型或分段时使用新 output_root；不把旧标注当作新结果。

## 查看

`artifacts/pretrain_noncot_20261011/index.html`：8 条完整 episode，全部动作时间轴、视频定位、逐段起中末帧、学生实际图片、完整 VLM 对话、VLA 原始动作序列、两种 prompt、原始标注、时间对齐及请求日志。

同名 `.zip` 是离线包。解压后直接打开 index.html，无需服务器；包内包含回放、截图、字体、数据和 teacher 输入拼图。`overview.jpg` 是真实观测总览。

`data/vlm/preview.jsonl`、`data/vla/preview.json` 汇总全部预览；`train`/`val` 另按原始来源划分，不能将预览集合直接混为训练 split。

`VALIDATION.json` 验证全轨迹覆盖、原始动作不变、guide/chunks、无 CoT、图像可读和来源时序；它不证明动作语义及边界都经过人工验收，也不把源 task 或 is_terminal 当作成功证据。
