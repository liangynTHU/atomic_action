# AgiBot 分成两半：两个脚本分别启动

已经准备好根目录的两个入口：

- `agibot_part1.sh`：处理第一半数据。
- `agibot_part2.sh`：处理第二半数据。

首次使用 GLM / TT Switch 的配置方法见 [AGIBOT_GLM_QUICKSTART.md](AGIBOT_GLM_QUICKSTART.md)。本文从服务已经配置好开始。

## 数据怎么分

2026-10-24 从当前 AgiBot 原始数据的 episode 索引读取到：

- 总计 **188 个 task，157,473 条 episode**。
- 第一组：**78,737 条**。
- 第二组：**78,736 条**。

按 task 编号、episode 编号排序，然后交替分配到两组。每个 task 中的 episode 也会分到两边，两组总条数最多相差 1。

两组没有重复，合起来覆盖完整原始 episode 清单。这里平分的是 **episode 数量**；视频长度和检查通过率不同，两边的运行时间及最终接受条数可能不同。

清单首次创建后固定保存，重复启动会验证并沿用它。数据处理继续使用原来的分割、GLM 标注和视觉复核流程。

当前只生成 **16D 双臂动作，按该动作范围视为没有底盘移动**。底盘位置、姿态和速度辅助字段不作为筛选条件；双臂末端、夹爪、时序和视频检查继续执行。

**本次规则更新使用新的 `_16d_split2` 输出父目录。** 旧的 `_split2` 输出已经冻结旧筛选代码和拒绝结果，需要按第 2 步重新设置 `AGIBOT_OUTPUT` 才能使用新规则。两份 episode 分组清单继续沿用。

## 1. 打开服务器终端

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source /jizhicfs/peterrao/miniconda3/bin/activate
export PATH="$PWD/tt-switch-cli/bin:$PATH"

tt-switch-cli start
tt-switch-cli status
```

TT Switch 已经运行时，`start` 会显示已有服务信息。

## 2. 设置两个任务共用的配置

默认模型仍为 `glm-5.3-flash-ioa`，推理强度为 `xhigh`，Token 从根目录 `.api_token` 读取。

两个脚本默认每组并发 `2`，合计 `4`。下面明确设置并发及本次输出目录，方便复制：

```bash
export AGIBOT_CONCURRENCY=2
export AGIBOT_OUTPUT=/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot/run_glm53_flash_xhigh_grounded_v7_16d_split2
```

**这里的 `AGIBOT_OUTPUT` 是两个任务共用的父目录。** 两个脚本自动使用：

```text
run_glm53_flash_xhigh_grounded_v7_16d_split2/
├── part1/    第一组的检查点、标注及训练结果
└── part2/    第二组的检查点、标注及训练结果
```

如果要沿用之前 `agibot_local.env` 中的 Python、API、模型等自定义设置，先执行 `source agibot_local.env`，再执行上面的两个 `export`。

续跑时使用原来的父目录。更换模型、提示词版本、批大小或数据来源时，使用新的父目录。

## 3. 查看分配信息并检查 API

```bash
bash agibot_part1.sh info
bash agibot_part2.sh info
```

`info` 会显示两组条数、清单位置和当前组的输出目录，不调用模型。

接着检查：

```bash
bash agibot_part1.sh check
bash agibot_part2.sh check
```

检查成功后，继续下一步。`check` 验证路径、依赖、配置和鉴权模型目录，不发起 GLM 推理。

## 4. 启动两个后台任务

在 **同一个服务器终端** 执行下面整块命令：

```bash
mkdir -p .runtime/agibot_split2

nohup bash agibot_part1.sh start >> .runtime/agibot_split2/part1.log 2>&1 &
echo $! > .runtime/agibot_split2/part1.pid

nohup bash agibot_part2.sh start >> .runtime/agibot_split2/part2.log 2>&1 &
echo $! > .runtime/agibot_split2/part2.pid
```

这会分别启动两组完整流程，产生真实 GLM 调用。每组各自准备清单、冻结源码、处理数据和导出结果。

查看两个进程：

```bash
ps -fp "$(cat .runtime/agibot_split2/part1.pid)"
ps -fp "$(cat .runtime/agibot_split2/part2.pid)"
```

后台启动后可以关闭这个终端。

### 想在前台看日志

也可以在两个服务器终端分别启动，每个终端先完成第 1、2 步的环境设置。

第一个终端：

```bash
bash agibot_part1.sh start
```

第二个终端：

```bash
bash agibot_part2.sh start
```

同一组同时运行一个任务。如果已经后台启动，就通过日志查看已有任务。

## 5. 看日志和进度

**默认每组每处理 1000 条 episode，输出一次进度汇总。** 前台直接显示，后台写入对应日志；本轮结束时也会输出汇总。

示例（数量仅用于展示格式）：

```text
[AgiBot 进度][part1][第1轮] 已处理 1000/78737 (1.27%) | 接受=850 | 原始数据舍弃=100 | 标注舍弃=40 | 待重试=10 | 处理错误=0 | 耗时=120.0分钟
```

“已处理”是当前轮次的 episode 数量，包含接受、舍弃、错误和复用缓存的数据。续跑或自动重试重新计数；“接受”表示本轮确认已经导出的数量。

只查看已经输出的进度汇总：

```bash
rg 'AgiBot 进度' .runtime/agibot_split2/part1.log
rg 'AgiBot 进度' .runtime/agibot_split2/part2.log
```

日志：

```bash
tail -n 50 -f .runtime/agibot_split2/part1.log
```

或：

```bash
tail -n 50 -f .runtime/agibot_split2/part2.log
```

按 Ctrl+C 退出日志查看，后台任务继续运行。

查看两组各自的完整进度：

```bash
bash agibot_part1.sh status
bash agibot_part2.sh status
```

如果在新终端执行，先重新进入项目目录，并设置第 2 步中相同的环境变量。

刚启动时如果提示 inventory 未提交，等准备数据清单的阶段完成后再查看。

## 6. 停止和续跑

前台运行：在对应终端按 **Ctrl+C**。

后台运行：先用第 4 步的 `ps` 命令核对进程号对应的 AgiBot 任务，再停止需要暂停的那一组：

```bash
kill -TERM "$(cat .runtime/agibot_split2/part1.pid)"
```

或：

```bash
kill -TERM "$(cat .runtime/agibot_split2/part2.pid)"
```

停止后，要继续哪一组，就重新执行该组的启动命令。同一份清单、同一个输出目录会自动恢复已提交阶段和已接受标注批次。

例如第一组前台续跑：

```bash
bash agibot_part1.sh start
```

第一组后台续跑：

```bash
nohup bash agibot_part1.sh start >> .runtime/agibot_split2/part1.log 2>&1 &
echo $! > .runtime/agibot_split2/part1.pid
```

## 7. 结果在哪里

先刷新两组进度：

```bash
bash agibot_part1.sh status
bash agibot_part2.sh status
```

两份正式结果索引：

```text
$AGIBOT_OUTPUT/part1/accepted.jsonl
$AGIBOT_OUTPUT/part2/accepted.jsonl
```

每行 `path` 指向该 episode 的正式训练导出目录。加载完整结果时，同时读取两份索引。

生成预览：

```bash
bash agibot_part1.sh review
bash agibot_part2.sh review
```

预览分别写入 `part1/preview/` 和 `part2/preview/`。浏览器访问方法见 [新手说明第 11 步](AGIBOT_GLM_QUICKSTART.md)。

## 清单文件与自定义设置

当前数据对应的固定清单保存在：

```text
.runtime/agibot_split2/cf4e3aab183916bd/selection/
├── part1.json
├── part2.json
├── summary.json
└── COMMIT.json
```

两个脚本启动时会自动准备或验证清单。只想准备清单，可以执行：

```bash
bash agibot_part1.sh split
```

`.runtime/` 被 Git 忽略。保留该目录可以直接沿用已生成清单；同样的原始索引也能按相同规则重新生成分配。

自定义规则：

- `AGIBOT_SOURCE`：两组共用的原始数据根目录。
- `AGIBOT_OUTPUT`：两组共用的输出父目录，脚本追加 `part1` / `part2`。
- `AGIBOT_SPLIT_DIR`：可选的清单保存目录，两组设置同一个值。跨机器使用时可设置为共享文件系统中的固定路径。
- `AGIBOT_CONCURRENCY`：**每组**并发数；设置为 `4` 时两组合计为 `8`。
- `AGIBOT_PROGRESS_EVERY`：每组进度汇总的 episode 间隔，默认 `1000`，可在启动前设为 `100` 等正整数。
- 其他模型和 API 的 `AGIBOT_*` 配置继续由原来的 `agibot.sh` 读取。

两个脚本管理数据来源、输出目录和选择范围，因此使用上述环境变量配置，不接受直接传入 `--source`、`--output`、`--selection`。

原始 episode 索引发生变化时，旧清单会报错，防止续跑时分配悄悄变化。开始新的数据版本时，使用新的 `AGIBOT_SPLIT_DIR` 和新的 `AGIBOT_OUTPUT`。

查看脚本帮助：

```bash
bash agibot_part1.sh --help
bash agibot_part2.sh --help
```
