# ABC-130k · 128 GPU 共享挂载执行提示词

将下面整段交给运行机器上的执行 agent。它是开发、部署和运行的任务指令，不是发给 Qwen 的标注 prompt。Qwen 使用代码中已经确定的 teacher prompt。

---

你需要在我分配的 128 张 GPU 上完成 ABC-130k 全量原子动作划分、非 CoT 子任务标注、训练数据导出与审计。我已经授权这次数据生产，包括补齐生产入口、部署本次节点上的模型、必要的小批验证、全量生成、重试及最终产物。请持续执行，不要仅给计划，也不要在小批成功后停下来再次询问是否全量。不要启动模型训练，不处理 OXE，不改写原始数据或其他实验输出。

## 1. 先识别输入、代码与运行资源

原始数据根目录：

```text
/apdcephfs_gy2/share_303214312/leoyizhang/datasets/ABC-130k
```

数据来自 `data/train/**/episode.mcap`、`data/val/**/episode.mcap`，部分 episode 带 `annotation.mcap`。随数据报告记录 train 129032 条、val 1671 条，合计 130703 条、3590.7 小时；请实际枚举并核对，不能将报告数字冒充已完成的扫描。稳定 episode ID 使用 split + 相对路径，不以扫描顺序或 Python hash() 作为身份。保留原 train/val 分组。

共享代码目录，直接从本机已经挂载的路径读取：

```text
/mnt/lyn/workspace/atomic_episode_segmentation
```

另一台服务器也已经挂载这些目录，不需要制作压缩包、传输源码、重新下载数据或模型，也不要要求我上传文件。直接核对共享路径。代码包含当前工作区未提交的修改和新增文件，不能只根据 Git HEAD 重新 clone。若某个节点挂载不可见，报告具体路径并处理实际挂载问题，不另造迁移包。

先读本机适用的 AGENTS.md，以及共享代码目录下的 `docs/PRETRAIN_NONCOT.md`、`configs/pretrain_noncot.soft_chunks_still.json`。采用 **2026-10-13 的软长度与静止段规则**，不回退到旧版“至少16帧”。

参考配置只是 4 ABC + 4 OXE 的小样本，包含旧输出复用路径：生产配置必须重新生成，只包含 ABC，去掉 `reuse_prepared_root` / `reuse_annotations_root`，换为本次实际共享输出目录和 endpoints。不能把旧样本配置直接当全量命令。

输出使用新的、所有数据 worker 可见的共享目录，例如在当前获准可写的共享父目录下创建 `abc130k_noncot_softchunks_128gpu_<实际日期时间>`。在本次清单中记录绝对路径。原始输入只读。

## 2. 固定运行版本，和其他开发隔离

直接以共享工作区当前内容为基线，在本次运行目录的独立工作副本中完成生产化修改，不直接修改其他任务正在使用的仓库。这里只复制所需源码用于固定运行版本，不压缩、不搬迁原始数据和模型。现有公共模块同时服务 OXE/ABC；我可能继续在原仓库开发 OXE。

完成修改并通过检查后，将本次代码、依赖版本、teacher/student prompt、配置和 endpoints 基线冻结到运行目录，计算内容 SHA256。所有 coordinator/worker 从这个版本启动，记录模块 `__file__` 路径和版本哈希，不能从原工作目录或旧 editable install 偷偷导入模块。

后续重启、数据 rank、补跑继续用同一冻结版本；语义参数、源文件或 prompt 变化必须成为明确的新版本。运行时吞吐配置和端点健康状态可以调整，但单独记日志，不能绕过语义版本校验。不要依赖“Python 已经导入，所以改代码无影响”。

## 3. 模型拓扑和部署

若本次分配的卡是此前使用的 H20 96GB，拓扑为：

```text
16 个节点 × 每节点 8 卡 = 128 卡
每节点 1 个 BF16 Qwen3.8-Flash-Next，TP8 + EP8
总计 16 个物理模型副本 / endpoints
```

模型名：`Qwen/Qwen3.8-Flash-Next`。

参考资源路径：

```text
权重 /apdcephfs_gy7/share_303588738/peterrao/hugmodels/Qwen3.8-Flash-Next
服务环境 /jizhicfs/peterrao/miniconda3/envs/sglang_qwen38_73a255
SGLang 源 /apdcephfs_gy7/share_303588738/peterrao/runtime_sources/sglang-qwen38-73a255206f916366
固定 commit 73a255206f916366c8d26d4022f82ddfb0ab558d
参考 CUDA /usr/local/cuda-13.2
```

以上是此前环境路径，必须在本机和每个节点实际验证，不能假定存在；环境不匹配时在本次独立目录修复。实际读取各节点 nvidia-smi，核对卡型、显存、占用和分配身份。不要杀掉别人的服务，也不要沿用参考配置中的旧 IP。若本次卡型不同，先验证合适的 TP8 部署，不把 H20 参数当作所有硬件的定论。

复用 `scripts/serve_model_node.sh` 和共享部署脚本 `/mnt/lyn/workspace/FastWAM/staging_atomic/deploy/serve_qwen38_node.sh`；将所用部署脚本也纳入本次版本记录。在本次每个分配节点启动一个服务，使用实际空闲端口。下例仅是服务命令模板，需填好本次冻结源码目录 `ABC_RUN_SOURCE` 和实际端口 `ABC_MODEL_PORT`：

```bash
export SERVE_SCRIPT=/mnt/lyn/workspace/FastWAM/staging_atomic/deploy/serve_qwen38_node.sh
export MODEL_PATH=/apdcephfs_gy7/share_303588738/peterrao/hugmodels/Qwen3.8-Flash-Next
export ENV_PREFIX=/jizhicfs/peterrao/miniconda3/envs/sglang_qwen38_73a255
export SOURCE_PATH=/apdcephfs_gy7/share_303588738/peterrao/runtime_sources/sglang-qwen38-73a255206f916366
export SERVED_MODEL_NAME=Qwen/Qwen3.8-Flash-Next
export PORT="$ABC_MODEL_PORT"
export MAX_MODEL_LEN=32768
export MEM_FRACTION_STATIC=0.85
export MAX_RUNNING_REQUESTS=16
bash "$ABC_RUN_SOURCE/scripts/serve_model_node.sh"
```

生成独立 endpoints 文件，恰好包含本次 16 个不同物理节点 URL，不使用 localhost 代替跨节点地址。所有节点 `/v1/models` 必须返回所需模型名。用真实 ABC 图片验证多模态请求，再验证实际一批 4 段、每段 5 张多视角拼图的请求；单图 health check 不等于生产请求验收。内网模型请求绕过外网代理。

客户端每 endpoint 初始并发 2，总 HTTP 并发 32；观察吞吐、延迟、失败率和显存后逐级升到每 endpoint 4 或 8。服务器 max-running-requests=16 是上限配置，不代表实测最佳客户端并发。TP rank、模型副本编号和数据 rank 要分别记录。

## 4. 保留当前分段与训练数据语义

复用以下实现作为语义基线，不另造一套不相容协议：

```text
atomic_seg/pretrain/adapters.py
atomic_seg/pretrain/segmentation.py
atomic_seg/pretrain/abc_still.py
atomic_seg/pretrain/length_policy.py
atomic_seg/pretrain/prompts.py
atomic_seg/pretrain/student_format.py
atomic_seg/pretrain/pipeline.py
atomic_seg/prompt_contract.py
```

ABC 原生 state/action 为14维：左 joint1～6、左 gripper、右 joint1～6、右 gripper。关节角是弧度，base-to-wrist；夹爪开度 0=closed、1=open。它不是 RoboTwin 的16维末端 xyz/quaternion。部分 MCAP 的 pose 为空，不能填零伪造：有完整位姿时用现有末端位姿分段核心，无位姿时用现有关节运动分段分支，并记录所用分支。

长度参数固定为：

```json
{
  "minimum_frames": null,
  "preferred_minimum_frames": 32,
  "preferred_maximum_frames": 64,
  "maximum_merged_frames": 128
}
```

这是同动作内部偏向较多 chunks 的软合并，不是16帧硬下限，也不是要求每段必须32/64帧。2026-10-21 起，相邻片段的左右臂动作标签必须分别相同才允许合并；闭爪与抬升、开爪与移动、移动与旋转等都保留独立边界。保护原始任务边界，末尾保留 hold 时不可仅为增加长度把它吞掉。128 是现有算法对新合并跨度的参数，不把已有更长完整原语裁断或丢弃。保留 `primitive_segments`、`motion_sequence`、原标注映射偏移、短段保留原因及 `length_audit.merge_policy`。旧混合动作结果须从原始 primitive 重新分段和标注，并按新边界重建 guide、chunks、历史与图片，写入新运行目录。

两条 ABC 分支都统一处理双臂均静止：开头合入下一动作，中间合入上一动作；末尾不足8行时吸收，足够长时允许一个末尾 hold；全静止 episode 保留单独标记的 hold，不当作成功示范。所有动作行保留。末端位姿核心自身做过的 still 处理也保留来源记录。

以当前 adapter 的原生 action 时间轴对齐各流；状态和学生当前/历史图像只能来自该 action 时刻或之前。读取每个相机的真实 H264/H265 codec 和 mono/stereo topic。不要假设所有消息数量相等；当前约束无法无损适配的记录进入明确错误/待复核清单，不擅自丢行、重采样或填造 action 来凑全量成功。保持现有250ms对齐限制，若真实数据触发大量异常，先给出实例再决定修复策略。

训练行顶层只有 `messages`、`images`；metadata 独立且逐行对应。student system 是连续段落，保留 RoboTwin 共同 planner 规则，仅说明真实14维坐标并移除 thinking 指令。user 的当前/历史状态、执行信息与图片顺序按 `student_format.py`。原始动作数值保持不变。

assistant 直接输出且只输出 `guide_action`、`primary_action_verb`、`sub_task`、`num_chunks` 四字段 JSON，无 CoT。

```text
guide_action = native_action[end] - native_state[start]
num_chunks = ceil((end - start + 1) / 8)
```

夹爪 delta=0 是开度不变，不代表闭合。不能把关节 delta 解释成厘米或仅靠正负判断 Cartesian 抬升/下降。padding 不算有效动作行，禁止重复图片或动作以增加 chunks。

## 5. Qwen 只生成 sub_task

使用代码中当前 ABC `teacher_system(..., grouped=True)`；显式关闭 thinking，温度0.1、输出上限2200 tokens、每批默认4个片段。每段提供5个去重的时序采样点，将实际多相机图像拼图；短段没有5个不同帧时只用可用帧。附原始 primitive 顺序、按臂 motion sequence、起终状态、粗粒度任务标注上下文。学生不能看到 teacher 的未来图。

Qwen 不写边界、guide 或 chunks。输出索引必须与请求完全一致；左右臂和动作动词校验沿用当前实现。原始请求参数、图片身份/哈希、原始响应、finish_reason、用量、耗时、重试和接受结果必须可追溯。

拒绝截断、额外推理正文、隐藏 reasoning、非法 JSON、少/多段、左右臂不匹配；有界重试后进入 review/error，不用模板冒充成功，不从隐藏 reasoning 截取答案。如果仅因 batch 上下文长度溢出，可以减小 batch 重试，记录原因，不能删除必要图像或改变语义。

## 6. 先实现生产 runner，不能直接扩容小样本脚本

当前 `generate_pretrain_noncot.py` / `pretrain.pipeline` 仍是样例流程：会累计所有 episode、生成全部 jobs/futures、逐帧 JPEG 落盘、生成完整可视化。这些路径不能原样跑130k。请在独立生产模块中复用语义函数，补齐以下能力：

1. 流式源 manifest 和按 episode 确定性分片。只启动一个总协调器；允许数据 rank 分担 episode，但同一输出 shard 只有一个 writer，不在16节点各跑一遍全量。
2. 有界解码队列、生成队列、内存和在途 episode 数。初始总计64个 CPU worker、每节点4个，根据 CPU/RAM/I/O 实测调整，不把所有数据载入内存。
3. 先读取 numeric stream 与相机时间戳、完成分段，再仅 materialize 学生当前/历史边界图和 teacher 所需采样图。为解码预测帧而读取必要视频包可以，但不把全部三路/四路相机的每帧输出为 JPEG。
4. 用本机临时目录做限额缓存；student 图片写入训练节点可达的持久存储，不能引用已清理的 `/tmp`。学生最终 images 必须保持可读取路径。图片按 episode/frame/camera 去重。状态/action 数组建议按 Parquet/NPZ 等保存并附稳定引用，避免全量重复嵌入巨量 JSON。
5. 采用有界大小的 VLM/metadata/manifest/日志分片，保留稳定 row/episode ID。VLM 仍然是 messages/images JSONL；如调整 VLA 外层容器做分片，明确 schema 和加载方式，并保留完整 native action 序列、帧区间及每段 target。
6. 请求/响应使用 append-only 分片 journal，避免每次请求一个小文件。记录失败及 review，不将未通过候选混入 accepted。部分片段未通过的 episode，在 manifest 中标明不完整并列出缺失段；不得冒充完整 episode。
7. 原子提交和断点恢复。重启只补未提交任务，验证代码/prompt/输入 fingerprint、结果去重及尾行完整性。已接受结果不重复生成；无法避免的“响应到达但未提交”重试也不能造成输出重复。shared FS 不适合的锁/WAL机制不要直接照搬。
8. endpoints 健康探测、单端点限流、故障退避和恢复。服务异常与标注内容不合格分开计数；不因一个坏端点丢失整组任务，不重启不属于本次的服务。
9. 数据处理与模型服务分别由调度器/可靠 supervisor 管理。保存本次 PID/job ID、日志、停止/恢复命令；不要依靠会随当前 shell 退出而终止的进程。

这些是要实现并验证的能力，不是已存在的参数。不要把未支持字段写进旧 JSON 就声称已经启用。原先公共 pretrain 路径并没有完整的运行代码版本锁，必须在生产 runner 中补上。

## 7. 检查通过后自动启动全量

按以下顺序推进，不额外等待一次全量确认：

1. 全量清点输入，检查16节点、128卡、共享路径、剩余空间和依赖。安装依赖用本次环境；数据CPU环境与SGLang环境分离。按本机 AGENTS.md 配置网络/conda，别修改系统 HOME。
2. 先做单节点真实多模态 smoke，再确认16个副本都能处理生产形状请求。
3. 运行必要的分段、静止、soft-length、14D、history、guide/chunks、图片对齐和恢复去重测试。至少包括不足16帧但有保护边界、双臂静止吸收、全静止、缺pose、非整数对齐、短尾段以及多相机顺序。
4. 先跑按 task、长度、codec、pose有无、annotation有无分层选取的约512个真实 episode。故意中断本次小批数据worker并恢复，验证无重复、无遗漏和已提交数据不变。为复现选样固定 seed/manifest，不能只取最前面的短样本。
5. 在共享输出目录保存8～12个完整episode的index及媒体，用此前页面结构展示，直接提供页面路径，不打包ZIP。原始失败、源坏文件、长度例外如实报告。结构、来源对齐及恢复检查通过，并且失败类别得到明确解释后自动继续全量。新的语义缺陷或不可恢复的基础设施问题必须修复或明确报告，不能降低验收标准凑通过。
6. 小批的正式接受结果若使用最终冻结配置可计入全量，避免无意义重复请求。全量仅自动处理新增/未提交记录。
7. 按固定间隔写 progress.json，持续更新处理速度和ETA；会话能保持工作时持续监控，不能仅启动一下就称任务完成。源错误/耗尽任务保留可复查清单和有界补跑入口。

## 8. 输出、进度和验收

至少交付：

```text
运行配置、冻结源码/依赖哈希、实际节点与GPU拓扑
source_manifest：每条源episode、split、文件身份、扫描/处理状态
action_segmentation：每条episode的边界、分支、primitive来源、still/length审计
data/vlm/{train,val}/...jsonl：accepted非CoT样本
metadata/{train,val}/...jsonl：与VLM逐行对应
data/vla/{train,val}/...：原生动作及目标，附明确schema
review/...：未通过样本/源异常/原因/补跑索引
logs/requests/...、logs/results/...：原始请求与结果journal
progress.json、THROUGHPUT.json、FINAL_REPORT.json
preview/index.html 及同目录媒体：抽样完整episode，直接使用共享路径
```

progress 包含发现/已处理/完成/失败episode、成功/失败/待复核片段、source/action行数、每个数据rank进度、16个端点健康、请求成功率/重试率、每分钟吞吐和ETA。至少统计每任务/分支的帧数及 num_chunks 分布，单列小于16帧、小于32帧、吸收still、全静止episode，确认软偏好是否达成，不通过改数字让分布“好看”。

完成检查必须包括：源episode数可核账；原始split保留；每条可解析轨迹完整覆盖、无缺口/重叠/丢行；每段 guide/chunks 与 native 数值一致；无 CoT；14维意义一致；当前/历史图片因果对齐；初始/中间无独立双臂still；持久图片路径有效；VLM/metadata/shard去重一致。

`processed=130703` 不等于 `accepted=130703`。必须区分全部源已处理、完整可训练episode、部分成功、待复核、源损坏/不支持等状态。不能要求所有源都通过而悄悄丢弃问题；FINAL_REPORT 给出每类数量和证据。

此前128卡3～7天只是基于少量实测的估算，不含生产runner改造时间、也不是承诺。请用512条分层小批和全量开始后的稳定吞吐重新估算。

请立即按以上任务执行。启动后先回报真实运行目录、16个端点验收结果、分层小批检查结果、全量启动命令/PID或job ID、进度位置及ETA；随后持续推进，最终回报完整数据、审计、失败清单和可视化路径。
