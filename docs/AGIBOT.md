# AgiBot: GLM-5.3-Flash API

2026-10-22: AgiBot 生产统一使用根目录 `agibot.sh`。已移除双组 GPU
启动器、节点发现和固定分片调度；不启动本地 Qwen/SGLang，不需要 GPU。
源 episode 筛选、动作分割、静止处理、16D 合同及 VLM/VLA 行格式保持不变。
已有旧输出及冻结源码不删除、不覆盖。

## 固定配置

| 项目 | 默认值 |
| --- | --- |
| 模型 | `glm-5.3-flash-ioa` |
| API Base URL | `http://127.0.0.1:15721/tencent/v1` |
| 令牌文件 | 仓库根目录 `.api_token` |
| 推理强度 | `xhigh` |
| 温度 | `0.1` |
| 输出上限 | `8192` tokens |
| 同时在途请求上限 | `4` |
| 每个标注请求的片段数 | `4` |
| 标注内容验证失败上限 | `3` |

模型名、图片能力和推理档位来自本机 TT Switch 通用模型目录。目录声明
不等于调用已验证；启动会先验证实际服务。

API 使用 Bearer 入站鉴权和 Chat Completions `reasoning_effort`，
不发送 Qwen 的 `chat_template_kwargs`。只解析 assistant `content` 中的最终
JSON，不从隐藏推理中提取答案。请求记录保留最终响应、模型、用量及是否
有隐藏推理，不保存隐藏推理正文或令牌。JSON 字段、片段索引、左右臂动词、
primitive 一致性和截断检查继续执行。

## 接入准备

`.api_token` 是 TT Switch 的通用 API 入站令牌，不是 CodeBuddy 上游密钥。
在 TT Switch 中需要分别配置：

1. 设置 -> 腾讯内网：启用服务并配置有效的 CodeBuddy 上游访问密钥。
2. 通用 API：配置入站 API Token，确保 `.api_token` 保存的是当前有效值。
3. 本机路由保持运行。端口 `8787` 是管理页面，`15721` 是模型代理。

2026-10-22 已按用户授权完成接入配置：

- 将 `.codebuddy_api_token` 中的上游凭据保存到 TT Switch，保留模型目录
  和各 Agent 的选模配置。
- 将 TT Switch 当前有效入站令牌同步到 `.api_token`，权限为 `0600`，
  没有重新生成服务令牌。
- 本机直连上游超时，出站网络代理已设为工作区提供的
  `http://star-proxy.oa.com:3128`。这是 TT Switch 全局出站代理，不是
  通用 API Base URL，不要把两者混填。

配置后的服务报告 `upstreamConfigured=true`、`tokenConfigured=true`、
`available=true`，没有阻塞项。真实 `bash agibot.sh probe` 已通过：
使用 `glm-5.3-flash-ioa`、`xhigh`、温度 `0.1`、输出上限 `8192`，
识别红色图片并返回 `{"color":"red"}`，耗时约 11.25 秒，上游报告
56 输入 tokens、8 输出 tokens。此请求验证图片输入、请求参数接受和
JSON 返回，不表示全量标注质量已验收；没有启动生产任务。

`agibot.sh` 不读取 `.codebuddy_api_token`，也不自动修改 TT Switch
设置或重新生成令牌。后续轮换凭据时，需要分别更新服务端上游密钥和
项目入站令牌文件。

## 启动

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
bash agibot.sh check
bash agibot.sh
```

`check` 只检查数据路径、依赖、配置及鉴权模型目录，不发起推理、不写
生产目录。`probe` 额外发起一次很小的红色图片请求，检查图片输入、推理参数
和 JSON 输出，可能产生少量 API 费用。默认 `start` 同样先执行这一检查，
通过后才枚举并处理全量数据。

```bash
bash agibot.sh probe
bash agibot.sh prepare
bash agibot.sh status
bash agibot.sh review
```

`prepare` 仅冻结源码和建立全量清单，不请求 API。默认 source:
`/apdcephfs_gy2/share_303214312/leoyizhang/datasets/agibot_beta_v3`。
默认 output:
`/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot/run_glm53_flash_xhigh_grounded_v7`。
默认 Python:
`/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python`。

输出使用新目录，不能混入旧 Qwen `run_v1`。默认全量处理，不会隐式选取预览。
小规模测试可以显式传入 `--selection /path/to/episodes.json`，文件必须是非空、
无重复的 `task_N/episode_000000` 字符串列表，并使用独立输出目录。

支持覆盖 `AGIBOT_SOURCE`、`AGIBOT_OUTPUT`、`AGIBOT_PYTHON`、
`AGIBOT_API_BASE`、`AGIBOT_API_TOKEN_FILE`、`AGIBOT_MODEL`、
`AGIBOT_REASONING_EFFORT`、`AGIBOT_TEMPERATURE`、`AGIBOT_MAX_TOKENS`、
`AGIBOT_CONCURRENCY` 和 `AGIBOT_BATCH_SIZE`，也支持同名 CLI 参数。
远程 API 必须 HTTPS；访问另一台机器的本地 TT Switch 时使用 SSH 隧道。

## 续跑和进度

前台运行，Ctrl+C 停止；正在执行的有限请求或视频检查可能先退出完成。
需要后台运行时：

```bash
nohup bash agibot.sh > agibot.log 2>&1 &
```

再次运行同一脚本会恢复已提交阶段和已接受标注批次。杀死进程后未落盘的
API 响应可能重算，已提交输出不会重复。全局 worker 锁防止同一输出目录
重复启动。并发数可在停止后调整，不改变 episode 身份。

首次准备后 `source/` 冻结代码和提示词，后续使用该版本继续生产。
修改模型、推理强度、温度、输出上限、批大小或选择范围需要新输出目录。
令牌从文件运行时读取，不写入配置或冻结目录，轮换令牌不影响已接受缓存。
HTTP 错误有界退避重试；重试耗尽保留 `retryable_error`，重新运行继续。
内容校验耗尽记为 `annotation_rejected`，整条 episode 不进入训练。

```text
source/                         冻结源码，不含 API 令牌
run_config.json                 规则、模型参数、依赖和版本哈希
inventory/episodes.jsonl         单一全量清单，无 group/rank 调度
inventory/index.jsonl            轻量进度索引
worker_progress.json            当前处理遍次进度
progress.json                   全量状态及舍弃原因
accepted.jsonl                  已完成的整条 episode 加载入口
episodes/<hash前两位>/<hash>/    分阶段检查点、图片和标注缓存
data/<split>/<hash前两位>/<hash>/
```

目录按稳定哈希前两位组织只是文件存储，不分配到不同作业。阶段采用
临时目录、SHA256、COMMIT 和原子发布，恢复以完整提交为准。
`status` 可在生产过程中刷新全量进度和 `accepted.jsonl`。

## 不变的数据规则

- 数值数组按 metadata 验证形状、长度、有限值，frame/index 连续、时间戳
  与 30 Hz 一致。缺失、重复、乱序、无效四元数整条舍弃。
- 任一时刻底盘有运动命令或有效姿态发生运动即舍弃，包括中间移动后回到
  起点。无法验证的底盘姿态记为 `base_unverifiable` 并舍弃。
- 底盘命令容差 `1e-6`，位移容差 `1e-4 m`，转动容差 `1e-4 rad`。
- 8 路相机完整解码，核对 PTS、30 Hz 和真实帧数，使用 LeRobot v3
  `from_timestamp` 偏移。不裁头尾、不补帧、不插值。
- 分割仍使用末端轨迹算法、双臂静止吸收和同动作内部的 32-64 动作行软
  长度偏好。任一臂动作标签变化保留边界，有意义短段保留。
- `chunk_size=8`。老师只生成 `sub_task`，不修改边界、guide 或 chunks。
  每段最多采样五个去重的时序观测，包含终点观测的三相机拼图。
- 学生只看当前及上一决策边界的头部和双手图，训练输出仍是非 CoT JSON。

## 标注质量迭代

2026-10-23：老师标注改为候选生成、独立视觉复核两个阶段。提示词要求先确认
物理手臂，再判断指间持物状态，最后描述当前片段的动作阶段。去除老师输入
中的粗粒度源阶段文字，保留上一段完整采样过程；明确空爪闭合不是抓到物体、
张开不是自动释放、工具移动不是自动执行擦拭或切割。

复核使用三路相机各自的时序图，额外记录左右手的起点、终点持物状态。
同一边界帧的 held/empty 明确矛盾会触发重看与有界重试；允许 unknown，
不把上一轮模型判断当真值。断点续跑恢复结构化边界观察，训练仍只写
`sub_task`，不混入复核字段。两阶段增加 API 请求量，且同模型复核仍可能
保留一致但错误的判断，不能替代看图验收。

方法、同样本对照结果及复现方式见 [AGIBOT_SEMANTIC_QUALITY.md](AGIBOT_SEMANTIC_QUALITY.md)。
复核时序图保留单帧 640×480，用三列两行按时间排列，并额外提供起点和终点
的完整头部图，避免超宽拼图影响工具细节。工具命名要求检查完整外形，不能仅凭把手颜色
换工具类别。输出模板辅助 JSON 格式；解析器只允许去掉完整的 JSON 外层围栏，
不提取隐藏推理、不修补损坏 JSON，所有结构和语义合同校验继续执行。
旧输出和冻结提示词保持不变；默认新目录使用 grounded_v7，尚未启动全量生产。
同一组5条 episode、56段的原版明确问题为15段，其中7段左右手错误；
grounded_v7 真实生成全部完成，逐段检查五个时序观测后未发现明确问题，
但保留3段待确认。这是反复调试的小样本，不能当作全量错误率或独立测试集结果。
原版与新版的数值、分割、学生输入和图片字节对照已通过；没有手工修改生成标签。

```text
S[t] = [left_xyz, left_quat_wxyz, left_gripper,
        right_xyz, right_quat_wxyz, right_gripper]
reference_action[t] = S[t+1]
guide_action = reference_action[end] - S[start]
num_chunks = ceil((end-start+1)/8)
gripper = (120 - measured_encoder) / (120 - 35)
```

N 个实测状态对应 N-1 个参考动作，保留终末观测。四元数从 xyzw 转 wxyz，
单位化并保持符号连续，guide 四元数项为分量差。位置保留源末端/法兰坐标系，
不伪造世界坐标变换。原始 native action 独立保留。
夹爪为固定近似编码开度参考值，不是毫米宽度标定，不做每条 min-max 或裁剪；
超出 `[-0.06, 1.06]` 的 episode 舍弃。

## 最终产物

每条完整接受的 episode 包含 `vlm.jsonl`、`metadata.jsonl`、
`reference.npz`、`vla.json`、`VALIDATION.json` 和 `COMMIT.json`。
VLM 每行顶层仍仅有 `messages`、`images`；metadata 逐行对应。
从根目录 `accepted.jsonl` 读取正式导出路径，不遍历 labels 或临时目录。
保留源 split，不额外创建隐含验证集。
