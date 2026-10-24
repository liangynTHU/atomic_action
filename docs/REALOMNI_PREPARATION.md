# RealOmni Open 整集检查、16D 参考动作与中断恢复

## 当前版本：首尾各最多裁剪 1 秒

用户确认放宽首尾规则后，新增 `trim_edges_1s`，`scripts/prepare_realomni.py` 默认采用该策略。旧 `strict` 仍可显式选择，旧结果目录保留。更改代码与规则后必须使用新运行目录；不会把新版判定写入旧 checkpoint。

- 两侧末端位姿、夹爪和相机是必需流。取共同有效时间区间，按源左手位姿时钟向内取真实观测；首尾实际删除量分别不得超过 1 秒。
- 因果对齐使用共同区间内、当前参考时刻之前最近的真实观测。为跨流采样相位而前移首个参考时刻所增加的裁剪也计入 1 秒限额。
- IMU 和未使用的相机标定只作诊断，不参与裁剪区间、分割、action 或 guide 的计算。
- 保留区间内的位姿、夹爪和相机时间间隙仍按既有阈值检查；不跨内部缺口拼接。视频逐帧验证保留区间，允许读取前面的编码包以解码，但不输出被裁掉的观测或额外动作。
- 裁剪后重新生成相邻实测参考 action、分割、16D guide、chunks 和历史起点。`crop` 记录源区间、共同区间、保留区间、首尾秒数、源帧号和每条流删去的行数。`coverage_scope=retained_interval`，不声称完整保留原 episode。

同一批 18 条重新运行，11 条通过并生成 **226 段、14,463 个有效参考动作行**；7 条拒绝：5 条关键流内部时间间隙、1 条首尾裁剪超限、1 条保留视频解码失败。运行目录 `artifacts/realomni_trim1s_run`，新版页面 `artifacts/realomni_trim1s_review/index.html`。

新版页面选出 6 条通过样本，显示保留区间、完整分段时间轴、两路原视频、每段起中末图片、16D guide、起点/目标状态、chunks、静止吸收和原始帧映射。默认播放保留区间，可勾选查看删除的首尾。页面使用数值运动标签，没有伪造 Qwen 子任务语义。

```bash
# 同一输出目录重复执行会校验并恢复；默认即本次 18 条样本
bash scripts/run_realomni_trim_examples.sh

# 自定义新运行目录和新预览目录
bash scripts/run_realomni_trim_examples.sh artifacts/realomni_trim_new_run artifacts/realomni_trim_new_review
```

固定样本清单为 `configs/realomni_examples.txt`；新版预览与浏览器检查入口分别为 `scripts/build_realomni_trim_review.py`、`scripts/check_realomni_trim_review.py`。下面保留首版 strict 策略及其历史结果说明，不将其拒绝数当作当前结果。

入口：`scripts/prepare_realomni.py`。当前范围是源数据检查、参考动作派生和原子动作分割；`segmented` 表示这些工作已提交，不表示已经完成模型子任务标注或整套训练数据生产。

## 筛选规则

按用户最后的澄清，**取消底盘筛选**。RealOmni 是人手采集；不根据双手轨迹、相机运动、IMU 或 `sim/robot_info.base_link` 推断底盘移动、采集者走动或底盘静止。每份检查报告都含 `base.filter_enabled=false`、`base.status=not_evaluated`。

只要任一必需数据流不能满足完整性检查，整个源 MCAP episode 进入 `rejected`，不保留局部有效窗口、不拼接、不插值、不用末帧补齐。

- 两侧必需流：末端位姿、相机压缩视频、夹爪磁编码器和 IMU；两侧相机标定必须存在且包含有效尺寸及 3×3 内参。
- 整文件检查 MCAP 首尾标识、消息计数及可用的 CRC；缺失、截断、非有限数值、无效末端四元数均不能通过。
- 名义频率分别为位姿/相机 30 Hz、夹爪 50 Hz、IMU 200 Hz。单流时间戳必须递增；相邻间隔超过 1.5 个名义周期即不通过。每条流相对所有必需流覆盖区间的首尾缺口最多 1.5 个自身名义周期。实际阈值写入运行版本，改变阈值必须新建输出目录。
- 传感器采样序号存在时检查连续性，支持 uint32 回绕。VIO 的 `sequence_num` 在真实文件中会重复，它不是位姿采样计数；位姿完整性使用时间戳和消息数，避免把重建更新序号当成丢帧证据。
- 在左手实测位姿时钟上做因果对齐：每个观测来自该时刻或之前，最大观测年龄为 1.5 个对应流名义周期。不能通过裁掉首帧、选择未来图像或向前填值来让 episode 通过。
- 数值与覆盖检查通过后，全量解码两路 H.264/H.265 视频；核对实际解码帧的时间戳、数量和损坏标记。前序检查已失败的 episode 直接整条排除，报告明确记录视频 `not_checked`，不声称检查过未解码的部分。

这里的“通过”指满足记录在案的完整性检查，不证明任务成功，也不能识别源文件从未记录且没有任何时间戳/序号证据的缺失。严格首尾规则可能导致大量真实 episode 被拒绝；拒绝原因保留原始 topic、缺口大小和异常位置。

文件系统临时故障、未支持的 schema 或依赖问题进入 `error`，下次重试；不会当成永久的数据缺失判定。

## Action / guide 合同

`realomni_umi` 使用左右各 8 维，共 16 维：

```text
[left_x, left_y, left_z, left_qw, left_qx, left_qy, left_qz, left_gripper,
 right_x, right_y, right_z, right_qw, right_qx, right_qy, right_qz, right_gripper]
```

位置沿用源 VIO 世界坐标，单位米；四元数为 wxyz，检查近似单位长度后归一化，并保持相邻符号连续。夹爪开度为原始米值除以 0.103，约 0=closed、1=open；不裁剪测量噪声。

对 N 个对齐后的实测状态 S，保留 N−1 个真实相邻转移：

```text
state[t] = S[t]
reference_action[t] = S[t+1]
guide_action[start:end] = reference_action[end] - state[start]
                        = S[end+1] - S[start]
valid_action_rows = end - start + 1
num_chunks = ceil(valid_action_rows / 8)
```

`reference_action` 是相邻实测状态派生的绝对参考目标，不是源文件提供的控制器命令。最后一个状态作为真实终点保留，不为它伪造额外 action。guide 四元数部分是分量差，不是 Euler 角或相对旋转；夹爪 delta=0 表示开度不变。

复用共享末端位姿分割、ABC 双臂 still 吸收和软长度策略：偏好 32–64 个有效动作行，没有 16 帧硬下限。2026-10-21 起只合并左右臂动作标签分别相同的相邻片段，任何一侧标签变化都保留边界；静止吸收仍单独保留来源。保留 primitive、still 吸收与长度审计。guide 保留 8 位小数，所有区间闭区间计数。代码指纹变化后须使用新运行目录，历史结果的段数不代表本次规则。

共享 student/teacher prompt、学生状态命名及导出审计已支持 `realomni_umi` 的 16D 表示。原有 `abc_yam` 14D 原生关节 adapter 继续使用自己的数据合同；这次没有把 ABC 的关节命令伪装成末端位姿，也没有声称旧 ABC adapter 已改为 16D。

## 持久化文件

```text
run_config.json                         代码、依赖、规则、源选择的版本身份
inventory/episodes.jsonl                 每条源 episode 的稳定 ID、路径、文件指纹
inventory/COMMIT.json                    清单提交记录
episode_status.jsonl                    每条 episode 的可读状态索引
progress.json                           发现数、状态计数、拒绝原因、已分割段数
episodes/<prefix>/<key>/status.json      每条 episode 最近状态、进程及错误信息
episodes/<prefix>/<key>/inspection.json  检查到哪个阶段及阶段结果
episodes/<prefix>/<key>/events.jsonl     处理、提交、错误事件
episodes/<prefix>/<key>/checked/
  decision.json                         accepted/rejected、逐项检查及源文件 SHA256
  episode.json                          通过样本的来源、对齐与数组映射
  trajectory.npz                        通过样本的完整实测状态、时间戳、原始数值流
  COMMIT.json                           本阶段全部文件哈希
episodes/<prefix>/<key>/segmented/
  segmentation.json                     完整分割、16D guide、chunks 与审计
  COMMIT.json                           分割阶段全部文件哈希
```

缺失样本只有质量判定，没有可用轨迹或分割结果。NPZ 中存完整实测状态；参考动作可以直接用 `measured_states[1:]` 读取，当前状态是 `measured_states[:-1]`，不重复存放三份相同的大数组。原始两侧未对齐数值也保留在 NPZ 中，字段映射位于 `episode.json.native_array_keys`。

## 强制中断后的行为

同一输出目录使用单 writer 锁，每条 episode 另有锁；进程被 SIGKILL 后由内核释放锁。阶段文件先写入临时目录并 fsync，生成含文件哈希的 COMMIT 后，目录原子重命名并同步父目录。

1. 检查未提交：保留最近的 `inspection.json` 供查看，重启重做整条检查；半写结果不视为通过。
2. 检查已提交且拒绝：校验输入身份和结果哈希后跳过，不再解码或分割。
3. 检查已提交且通过：从已提交轨迹继续分割，不重做质量判断。
4. 分割已提交：校验后跳过；即使在目录提交后、`status.json` 更新前被 kill，也会从 COMMIT 恢复为 `segmented`。
5. 半写 JSONL 的最后一行另存为 `.torn.*` 后截除；完整行损坏会报错，不静默丢失历史记录。

恢复时还会检查源内容 SHA256，即使源文件大小和修改时间相同也能发现替换。源文件、代码、依赖或语义参数改变不会混用旧结果。重启会重新读取源文件计算哈希，但不会重复已提交的解码和分割。

`progress.json` 在 episode 间约每 30 秒重建一次，长 episode 处理中可能滞后；每条 episode 的 `status.json` / `inspection.json` 是更细的进度依据。强制中断后可用 `--report-only` 重建汇总。

## 运行与恢复

本工作区可用的 CPU 环境：

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/prepare_realomni.py \
  --output-root artifacts/realomni_prepare \
  --max-episodes 100
```

首次枚举源清单；`--max-episodes` 只限制本次处理的未完成 episode 数。重复同一条命令自动继续，去掉上限则处理完整已冻结清单。可以重复 `--episode '<源相对路径>.mcap'` 固定一个小样本集合；固定集合属于运行版本的一部分。

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/prepare_realomni.py \
  --output-root artifacts/realomni_prepare --report-only
```

本次真实检查产物在 `artifacts/realomni_strict_smoke`：6 个类别样本，6 条拒绝，原因均包含时间间隙或首尾覆盖不足，另有 2 条缺少相机标定。没有把这些真实拒绝样本当作成功分割案例；完整通过与分割分支用实际编码的合成 protobuf/MCAP/H.264 文件验证。没有启动全量或 Qwen 标注。

测试入口：

```bash
PYTHONPATH=.runtime/pretrain_deps /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  -m pytest tests/test_realomni.py tests/test_pretrain_noncot.py \
  tests/test_segmentation.py tests/test_prompt_contract.py -q
```

包括必需流缺失/中间缺口/首尾缺口、坏四元数、时间倒序、因果对齐、16D 目标与终点、共享导出、坏视频、MCAP 截断，以及 6 个检查和分割提交位置的真实 SIGKILL 恢复测试。

## 真实结果预览与复现脚本

`artifacts/realomni_examples_review/index.html` 展示 6 条真实 episode 的检查结果，其中 5 条提供完整双路源视频和起中末截图；第 6 条 Stage 3 样本在预览时检出视频解码失败，页面明确显示错误，不播放残缺视频。各样本都展示首尾覆盖、间隙位置、具体阈值与原始判定。`examples.json` 是页面完整数据，`all_results.jsonl` 汇总 18 条真实样本的检查结果（原有 6 条，加其他目录补查的 12 条）。18 条均被当前严格规则拒绝，没有把拒绝轨迹用于分割或输出 guide/chunks。预览转码只缩小分辨率，时间轴保留源时间戳。

一键复现原有 6 条检查和预览：

```bash
bash scripts/run_realomni_examples.sh all
```

默认写入 `artifacts/realomni_examples_run`，网页在该目录的 `review/index.html`。指定输出目录时使用第二个位置参数；重复运行同一命令会继续质量检查与分割，已生成的完整预览会复用。只查看状态：

```bash
bash scripts/run_realomni_examples.sh status artifacts/realomni_examples_run
```

脚本支持 `all / check / review / status`；Python 环境可通过 `REALOMNI_PYTHON` 环境变量指定。各阶段实际入口为：

- `scripts/prepare_realomni.py`：检查、分割、恢复。
- `scripts/build_realomni_review.py`：从已提交结果生成视频和离线网页，不改变质量判定。
- `scripts/check_realomni_review.py`：可选 Chromium 验证，需要 Playwright 和已安装浏览器。

只对已有结果生成本次 18 条检查 / 6 条视频的预览：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=.runtime/pretrain_deps \
  /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python scripts/build_realomni_review.py \
  --run-root artifacts/realomni_strict_smoke \
  --run-root artifacts/realomni_additional_examples \
  --output-root artifacts/realomni_examples_review --limit 6
```
