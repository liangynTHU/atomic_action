# AgiBot 使用 GLM：从 TT Switch 安装到启动数据处理

适用项目：`/mnt/lyn/workspace/atomic_episode_segmentation`  
命令核对日期：2026-10-24  
使用模型：`glm-5.3-flash-ioa`

这份说明按操作顺序写。第一次使用，从第 1 步开始；已经配置好，直接看最后的“下次启动”。

**需要平分数据、两个任务分别启动时**，完成本文的服务配置后，按照 [AgiBot 两组启动说明](AGIBOT_SPLIT_START.md) 使用 `agibot_part1.sh` 和 `agibot_part2.sh`。两组各自使用固定清单和独立输出目录。

## 先知道自己要做什么

整个流程是：

```text
服务器启动 TT Switch CLI
  ↓
浏览器打开 TT Switch 管理页面
  ↓
配置 CodeBuddy 上游密钥和网络代理
  ↓
把“通用 API”的 Token 保存到项目
  ↓
检查 GLM 能否读取图片
  ↓
先处理两条 AgiBot 数据
  ↓
查看结果，再处理全量数据
```

AgiBot 的启动入口是根目录的 `agibot.sh`。它读取 AgiBot 原始数据，检查数值和视频，按运动轨迹分割，再调用 GLM 生成 `sub_task` 并进行独立视觉复核，最后导出训练数据。

**标注可以包含图像中看得清的物体名称、颜色，以及接触、持物和动作描述。** 描述需要与当前片段的实际图像一致，并区分物理左臂和右臂。GLM 负责文字标注，分割边界和数值动作由程序计算。

GLM 在 API 服务端运行，这个启动方式无需本地 GPU。最终学生训练输出使用直接 JSON，推理强度 `xhigh` 用于老师请求。

当前双臂参考动作固定为 **16D**：左右臂各为位置 3 维、姿态四元数 4 维、夹爪 1 维。按这个动作范围视为没有底盘移动，底盘辅助字段不作为拒绝条件；双臂数值和视频仍需通过检查。已有使用旧底盘筛选规则的任务，换新输出目录后才能采用新规则。

### 命令在哪里执行

- **【服务器终端】**：登录存放代码和数据的服务器后执行。TT Switch 和 AgiBot 使用同一个系统用户。
- **【电脑终端】**：在你自己的电脑上执行，用于建立 SSH 转发。
- **【电脑浏览器】**：在你自己的电脑上打开管理网页或结果预览。

复制代码块时，整块复制；不要额外输入终端提示符 `$`。

下面使用当前服务器已有的项目、Python 和数据路径。换服务器时，需要替换第 5 步中的对应路径。

## 1. 在服务器上找到并启动 TT Switch CLI

**【服务器终端】**

进入项目目录：

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
```

### 1.1 当前服务器：使用已有安装

当前工作区的 `tt-switch-cli/` 已包含完整安装。执行：

```bash
export PATH="$PWD/tt-switch-cli/bin:$PATH"
node --version
tt-switch-cli version
```

Node.js 要求为 **18 或更高版本**。当前本地 TT Switch CLI 核对版本为 `0.9.54`。

如果能够输出版本号，继续第 1.3 步。

### 1.2 找不到安装时：先安装

`tt-switch-cli/` 是本地安装目录，Git 克隆出来的项目可能没有它。

先检查：

```bash
node --version
npm --version
```

如果提示 `command not found`，需要先在服务器准备 Node.js 18 或更高版本及 npm。

有 Node.js 和 npm 后，在当前工作区可以通过下面的代理访问安装源：

```bash
export http_proxy=http://star-proxy.oa.com:3128
export https_proxy=http://star-proxy.oa.com:3128
export no_proxy=localhost,127.0.0.1,::1
export NO_PROXY="$no_proxy"

npm install -g tt-switch-cli \
  --registry=https://npm.cnb.woa.com/tt-switch/tt-switch-cli/-/packages/

tt-switch-cli version
```

安装需要下载对应平台的可执行文件，保留 npm 的 optional dependencies。

如果服务器不能下载安装源，但你有完整的 Linux x64 安装包，可以把包放在项目根目录，然后执行：

```bash
tar -xzf tt-switch-cli-0.9.54-linux-x64.tgz
export PATH="$PWD/tt-switch-cli/bin:$PATH"
tt-switch-cli version
```

这个完整包适用于 Linux x64；只有 `tt-switch-cli-0.9.54.tgz` 的主包仍需要安装平台依赖。

### 1.3 启动服务

```bash
tt-switch-cli start
tt-switch-cli status
```

看到 `running`，表示 TT Switch 后台服务已经启动。

记住两个端口：

- **8787**：浏览器管理页面。
- **15721**：程序请求模型的代理端口。

TT Switch 后台服务在关闭终端后继续运行。服务器重启后，需要重新执行 `tt-switch-cli start`。

## 2. 在电脑上打开管理页面

### 2.1 代码在远程服务器：建立 SSH 转发

**【电脑终端】**

下面的 `你的用户名@服务器地址`，替换为你平时登录这台服务器使用的地址；已经配置 SSH 别名的，也可以填别名。

```bash
ssh -N \
  -L 127.0.0.1:18787:127.0.0.1:8787 \
  你的用户名@服务器地址
```

登录成功后，这个终端没有新输出通常是正常的。**保持它开着。**

**【电脑浏览器】**

打开：

```text
http://127.0.0.1:18787
```

这里电脑的 `18787` 被转发到服务器的 `8787`。

如果电脑的 `18787` 被占用，可以把 SSH 命令中的第一个 `18787` 和浏览器地址一起改为 `18788`。

### 2.2 浏览器就在运行 TT Switch 的机器上

直接打开：

```text
http://127.0.0.1:8787
```

后面的 AgiBot 命令都在服务器执行。AgiBot 的模型地址使用服务器自己的 `127.0.0.1:15721`。

## 3. 在网页中配置上游服务、代理和模型

**【TT Switch 管理页面】**

### 3.1 填入 CodeBuddy 上游密钥

先准备一个有 GLM 调用权限的 **CodeBuddy 上游 API Token / API Key**。

按下面的顺序操作：

1. 打开 **设置 → 腾讯内网**。
2. 在 **内网连接** 中启用腾讯内网服务。
3. 保留适用于你的 CodeBuddy 服务的上游地址；首次使用当前内网服务时可沿用默认地址。
4. 在 **API Key / Token** 位置填入你的 CodeBuddy 上游密钥。
5. 点击保存。

也可以使用 **设置 → 腾讯内网 → 腾讯内网快速设置向导 → 开始配置**。向导最后要点击 **完成**，配置才会应用。

AgiBot 使用“通用 API”入口。向导可以只配置内网连接和模型目录，Agent 选择可以留空。

已有有效的上游密钥时，直接沿用即可。历史文件 `.codebuddy_api_token` 可以用于人工配置上游凭据；AgiBot 启动脚本读取的令牌文件见第 6 步。

### 3.2 配置服务器访问上游的网络代理

当前工作区访问上游服务时使用：

```text
http://star-proxy.oa.com:3128
```

在网页中：

1. 打开 **设置 → 路由**。
2. 找到 **出站代理**。
3. 选择 HTTP 代理，填入上面的地址。如果页面分别填写主机和端口，主机填 `star-proxy.oa.com`，端口填 `3128`。
4. 按页面提供的方式测试，并保存。
5. 确认本机路由正在运行，模型代理端口保持 `15721`。

如果这组代理不可用，当前工作区还提供：

```text
http://9.21.0.122:11113
```

按服务器实际网络情况选择一组。代理地址填写在 TT Switch 的“出站代理”设置中。

终端的 `http_proxy` / `https_proxy` 用于该终端启动的程序；已运行的 TT Switch 服务应在网页中核对并保存自己的出站配置。

### 3.3 确认完整模型 ID

打开左侧 **通用API**，等待模型目录加载完成；需要时刷新或更新模型目录。

找到完整模型 ID：

```text
glm-5.3-flash-ioa
```

本说明使用：

```text
模型：glm-5.3-flash-ioa
推理强度：xhigh
图片输入：需要支持
```

后面的 `check` 和 `probe` 会分别检查目录和实际图片请求。实际请求的模型由第 5 步的项目配置指定。

## 4. 准备运行 AgiBot 的 Python

**【服务器终端】**

回到项目目录，使用当前服务器已有的环境：

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source /jizhicfs/peterrao/miniconda3/bin/activate

export AGIBOT_PYTHON=/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python

"$AGIBOT_PYTHON" -c 'import numpy, scipy, pyarrow, av, PIL, h5py; print("AgiBot 依赖检查通过")'
```

看到 `AgiBot 依赖检查通过`，继续下一步。

如果出现 `ModuleNotFoundError`，可在你用于该项目的 Python 环境中安装项目依赖：

```bash
"$AGIBOT_PYTHON" -m pip install -e '.[pretrain]'
```

然后重新执行上面的依赖检查。已经开始生产的任务使用固定依赖版本，续跑时保持原来的 Python 环境。

## 5. 保存一份固定的启动配置

**【服务器终端，项目根目录】**

下面的命令创建 `agibot_local.env`。它保存数据路径和模型参数，方便下次直接加载。

**首次运行先给输出目录起一个新名字。** 下面使用 `16d_20261025_01` 作为例子；如果这个目录已经用于其他任务，把末尾改为 `16d_20261025_02` 或你自己的名字。

```bash
cat > agibot_local.env <<'EOF'
export AGIBOT_PYTHON=/jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python
export AGIBOT_SOURCE=/apdcephfs_gy2/share_303214312/leoyizhang/datasets/agibot_beta_v3
export AGIBOT_OUTPUT=/apdcephfs_gy7/share_303588738/peterrao/pretrain_data/Agibot/run_glm53_flash_xhigh_16d_20261025_01

export AGIBOT_API_BASE=http://127.0.0.1:15721/tencent/v1
export AGIBOT_API_TOKEN_FILE=/mnt/lyn/workspace/atomic_episode_segmentation/.api_token
export AGIBOT_MODEL=glm-5.3-flash-ioa

export AGIBOT_REASONING_EFFORT=xhigh
export AGIBOT_TEMPERATURE=0.1
export AGIBOT_MAX_TOKENS=8192
export AGIBOT_CONCURRENCY=4
export AGIBOT_BATCH_SIZE=4
EOF

source agibot_local.env
```

你主要关心这些配置：

- `AGIBOT_SOURCE`：原始 AgiBot 数据目录。当前服务器的数据已经挂载在这个位置。
- `AGIBOT_OUTPUT`：本次生成结果的目录。续跑时沿用同一个路径。
- `AGIBOT_API_BASE`：服务器上 TT Switch 的通用 API 地址，保留 `/tencent/v1`。
- `AGIBOT_CONCURRENCY`：同时处理的并发数，先用 `4`。
- `AGIBOT_BATCH_SIZE`：每次标注请求包含的片段数，先用 `4`。

需要改配置时，用编辑器打开 `agibot_local.env`，修改后重新执行 `source agibot_local.env`。

每次新开服务器终端，都需要重新加载这份配置。

## 6. 把“通用 API”的 Token 保存到项目

**【TT Switch 管理页面】**

打开左侧 **通用API**，找到页面的 **API Token**。

两种 Token 各有用途：

- **CodeBuddy 上游密钥**：第 3 步保存在 TT Switch 中，用于 TT Switch 访问模型服务。
- **通用 API Token**：这一页提供，保存在项目 `.api_token` 中，用于 AgiBot 访问 TT Switch。

优先复制当前有效的通用 API Token。首次使用、页面要求生成时，再生成一个。**重新生成会使旧 Token 立即失效，使用旧值的其他程序也需要更新。**

**【服务器终端，项目根目录】**

如果项目已有与网页一致的 `.api_token`，可以直接进入第 7 步。

需要保存或更新时，执行下面这段：

```bash
"$AGIBOT_PYTHON" - <<'PY'
from getpass import getpass
from pathlib import Path
import os

token = getpass("粘贴 TT Switch 通用 API Token，然后按回车（输入不显示）: ").strip()
if not token or any(ch.isspace() for ch in token):
    raise SystemExit("Token 格式不对，请重新复制通用 API 页面中的完整 Token。")

path = Path(os.environ["AGIBOT_API_TOKEN_FILE"])
path.touch(mode=0o600, exist_ok=True)
path.chmod(0o600)
path.write_text(token + "\n", encoding="utf-8")
print("已保存 .api_token，权限为 600。")
PY
```

出现提示后，粘贴 Token，再按回车。输入内容不可见是正常现象。

文件内容应为单个完整 Token。粘贴时只复制 Token 本身，省略 `Bearer ` 前缀和引号。

上述方式不会把 Token 放进命令历史。`.api_token` 已被项目的 Git 忽略规则排除。

## 7. 检查配置，再验证一次 GLM 图片请求

**【服务器终端】**

先执行：

```bash
bash agibot.sh check
```

它检查数据路径、Python 依赖、配置和带鉴权的模型目录。这个命令不发起模型推理。

成功时，输出包含：

```json
{
  "model": "glm-5.3-flash-ioa",
  "api_base": "http://127.0.0.1:15721/tencent/v1",
  "reasoning_effort": "xhigh",
  "catalog_accessible": true
}
```

然后执行实际图片检查：

```bash
bash agibot.sh probe
```

`probe` 会给 GLM 发一张很小的红色图片，验证图片输入、请求参数和 JSON 返回。**这一步会产生一次实际模型调用和相应用量。**

成功时，输出包含 `"passed": true`，返回内容中的 `color` 为 `red`。

如果任一步报错，先按后面的“常见问题”处理，修好后再执行这一节。

## 8. 建议先处理两条数据

这一步会真实运行数据流程和 GLM 标注。试跑使用独立输出目录：

```text
第 5 步配置的 AGIBOT_OUTPUT + _smoke
```

### 8.1 从真实数据中生成两条 episode 的清单

**【服务器终端，项目根目录】**

执行下面这段。程序会读取数据中的 episode 索引，创建 `agibot_smoke_episodes.json`：

```bash
"$AGIBOT_PYTHON" - <<'PY'
from pathlib import Path
import json
import os
import pyarrow.parquet as pq

selection_path = Path("agibot_smoke_episodes.json")
if selection_path.exists():
    raise SystemExit("试跑清单已存在。沿用清单即可，继续执行第 8.2 步。")

root = Path(os.environ["AGIBOT_SOURCE"])
selected = []
for task in sorted(root.glob("task_*")):
    for shard in sorted((task / "meta/episodes").glob("chunk-*/*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=32, columns=["episode_index"]):
            for episode_index in batch.column(0).to_pylist():
                identity = f"{task.name}/episode_{int(episode_index):06d}"
                if identity not in selected:
                    selected.append(identity)
                if len(selected) == 2:
                    break
            if len(selected) == 2:
                break
        if len(selected) == 2:
            break
    if len(selected) == 2:
        break

if len(selected) != 2:
    raise SystemExit("没有找到两条 episode，请检查 AGIBOT_SOURCE 和 meta/episodes 目录。")

selection_path.write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
print("试跑清单已创建：")
print(json.dumps(selected, indent=2))
PY
```

清单使用真实的 `task_N/episode_000000` 身份。之后继续这次试跑时，保留这份清单。

### 8.2 启动两条数据的完整流程

```bash
bash agibot.sh start \
  --output "${AGIBOT_OUTPUT}_smoke" \
  --selection "$PWD/agibot_smoke_episodes.json"
```

等命令结束后，查看试跑进度：

```bash
bash agibot.sh status --output "${AGIBOT_OUTPUT}_smoke"
```

`complete` 表示该 episode 已完整导出。数值或视频检查不通过、文字标注校验不通过的 episode 会记录舍弃原因。

选出的两条数据也可能都被筛掉。此时查看 `rejection_reasons`；第 7 步的 `probe` 用于判断实际模型连接是否通过。

第 11 步介绍如何打开结果预览。检查物体名称、颜色、左右臂、接触和持物描述是否符合视频后，再决定启动全量。

试跑中断后，重新执行同一条第 8.2 步命令即可续跑。

## 9. 启动全量 AgiBot 数据处理

**【服务器终端】**

先加载第 5 步保存的配置：

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source agibot_local.env
```

**下面的启动命令会处理 `AGIBOT_SOURCE` 下的全量 episode。**

### 9.1 前台运行：适合直接观察日志

```bash
bash agibot.sh start
```

根目录直接执行 `bash agibot.sh` 也等价于启动。

启动会先检查 API，并进行图片探测，然后冻结本次源码、建立数据清单、处理数据和导出结果。准备阶段可能先输出 `inventory task_...`。

前台运行时保持终端连接。需要停止，按 **Ctrl+C**；正在执行的有限请求或视频检查可能要先结束。

### 9.2 后台运行：适合长时间任务

如果已经启动了前台任务，先停止它，再使用后台方式。

```bash
mkdir -p "$(dirname "$AGIBOT_OUTPUT")"

nohup bash agibot.sh start > "${AGIBOT_OUTPUT}.log" 2>&1 &
echo $! > "${AGIBOT_OUTPUT}.pid"

cat "${AGIBOT_OUTPUT}.pid"
```

最后输出的是本次后台任务的进程号。

日志位置：

```text
${AGIBOT_OUTPUT}.log
```

进程号记录位置：

```text
${AGIBOT_OUTPUT}.pid
```

日志和进程号文件位于输出目录的旁边。后台启动后，可以关闭这个服务器终端；服务器重启会终止任务。

## 10. 看进度、停止和续跑

### 10.1 查看全量进度

**【另一个服务器终端】**

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source agibot_local.env

bash agibot.sh status
```

常见状态：

- `pending`：还未处理。
- `complete`：整条 episode 已通过并导出。
- `rejected`：原始数据检查不通过。
- `annotation_rejected`：标注在规定的重试次数内仍未通过校验。
- `retryable_error`：API 等可重试错误，需要继续运行。

命令会刷新 `progress.json` 和已完成的 `accepted.jsonl`。如果刚启动时提示清单尚未提交，等启动日志中的 inventory 阶段完成后再查看。

查看后台日志：

```bash
tail -n 50 -f "${AGIBOT_OUTPUT}.log"
```

按 Ctrl+C 退出日志查看，后台任务继续运行。

### 10.2 停止任务

前台任务：在运行窗口按 **Ctrl+C**。

后台任务：先核对进程号对应的命令：

```bash
ps -fp "$(cat "${AGIBOT_OUTPUT}.pid")"
```

确认它是本次 AgiBot 进程后，执行：

```bash
kill -TERM "$(cat "${AGIBOT_OUTPUT}.pid")"
```

后台停止时，未落盘的模型响应在续跑时可能重新请求；已提交阶段和已接受的标注批次可以恢复。

### 10.3 继续同一次任务

重新打开服务器终端，加载同一份配置：

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source agibot_local.env

bash agibot.sh start
```

需要继续后台运行，则重新执行第 9.2 步的后台启动命令。那里使用 `>`，会从新的一份日志开始记录；需要保留上一份日志时，先给它改名。

**续跑的关键是同一个输出目录、同一套模型参数和同一个选择范围。** 程序自动恢复，无需额外填写 `--resume`。

修改模型、推理强度、温度、输出 token 上限、批大小、数据选择范围或依赖版本时，使用新的输出目录。并发数可以在任务停止后调整。

首次准备时，代码和提示词保存在输出目录的 `source/`。同一输出目录后续沿用该冻结版本；希望使用改过的代码时，新建一次运行。

同一个输出目录同时运行一个任务。

## 11. 用浏览器查看结果

### 11.1 生成预览

**【服务器终端，已加载 agibot_local.env】**

查看全量任务时，执行：

```bash
bash agibot.sh review
export AGIBOT_PREVIEW_DIR="$AGIBOT_OUTPUT/preview"
```

如果查看第 8 步的两条试跑数据，改为执行下面这一组：

```bash
bash agibot.sh review --output "${AGIBOT_OUTPUT}_smoke"
export AGIBOT_PREVIEW_DIR="${AGIBOT_OUTPUT}_smoke/preview"
```

选择上面的一组即可。预览生成会读取视频，可能需要等一会儿。

默认预览最多包含 **8 条已有检查状态的 episode**，可能包含舍弃的数据。它用于查看样例；全量统计通过 `status` 查看。

### 11.2 在服务器启动预览网页

接着在同一个服务器终端执行：

```bash
"$AGIBOT_PYTHON" -m http.server 18080 \
  --bind 127.0.0.1 \
  --directory "$AGIBOT_PREVIEW_DIR"
```

保持这个终端开着。

### 11.3 从电脑访问预览

**【电脑的另一个终端】**

```bash
ssh -N \
  -L 127.0.0.1:18080:127.0.0.1:18080 \
  你的用户名@服务器地址
```

**【电脑浏览器】**

打开：

```text
http://127.0.0.1:18080
```

预览中结合头部、左右手视频与分割标签，检查：

- 左右臂有没有写反。
- 物体名称和颜色是否看得见、是否与实际物体一致。
- 靠近、接触、抓持、松开有没有与当前片段混淆。
- 工具运动有没有被误写成已经擦拭、切割或完成放置。

关闭预览服务时，在运行 `http.server` 的服务器终端按 Ctrl+C。

## 12. 最终训练数据在哪里

先执行一次：

```bash
bash agibot.sh status
```

然后查看已完成的 episode 索引：

```bash
head -n 3 "$AGIBOT_OUTPUT/accepted.jsonl"
```

每行包含 episode 身份、源 split、正式导出路径 `path` 和片段数 `segments`。**训练数据从这个索引中的 `path` 查找。**

主要目录：

```text
本次 AGIBOT_OUTPUT/
├── source/                  本次冻结的代码
├── run_config.json          本次配置和版本
├── inventory/               本次 episode 清单
├── progress.json            全量进度和舍弃原因
├── worker_progress.json     当前处理遍次的进度
├── accepted.jsonl           已完整导出的 episode 索引
├── episodes/                检查点、图片和标注缓存
├── preview/                 执行 review 后生成的网页
└── data/<split>/<前两位哈希>/<episode哈希>/
    ├── vlm.jsonl
    ├── metadata.jsonl
    ├── reference.npz
    ├── vla.json
    ├── VALIDATION.json
    └── COMMIT.json
```

`vlm.jsonl` 是学生 VLM 训练记录，顶层字段为 `messages` 和 `images`。图片记录使用输出中的实际图片路径，后续使用数据时保留对应图片和目录。

通过检查的 episode 整条导出。原始数据或标注检查未通过的 episode 会记录状态和原因。

## 常见问题

### `tt-switch-cli: command not found`

当前服务器已有本地安装时，在项目根目录执行：

```bash
export PATH="$PWD/tt-switch-cli/bin:$PATH"
tt-switch-cli version
```

若目录不存在，按第 1.2 步安装。

### 网页打不开

先在服务器执行：

```bash
tt-switch-cli status
```

确认服务为 `running`，再检查电脑上的 SSH 转发终端是否还开着。

远程访问时，浏览器使用电脑入口 `18787`；如果改了转发端口，浏览器也使用改后的端口。

### HTTP 401 或 403

按顺序检查：

1. 项目 `.api_token` 是否对应“通用API”页面的当前 Token。
2. **设置 → 腾讯内网** 中的上游密钥是否有效。
3. 账号是否有该 GLM 模型的调用权限。

更新项目 Token 按第 6 步操作，然后重新执行 `check` 和 `probe`。

### HTTP 404、模型不存在，或不支持 `xhigh`

检查 `agibot_local.env`：

```text
AGIBOT_API_BASE=http://127.0.0.1:15721/tencent/v1
AGIBOT_MODEL=glm-5.3-flash-ioa
AGIBOT_REASONING_EFFORT=xhigh
```

API Base 填到 `/tencent/v1`。`/models` 和 `/chat/completions` 由程序添加。

在通用 API 页面刷新模型目录，核对完整 ID 和推理档位。确实需要换模型或档位时，也给已有任务换一个新输出目录。

### 连接超时、API connection failed

先确认 TT Switch 和本机路由正在运行，再检查 **设置 → 路由 → 出站代理** 是否已保存、代理是否可用，以及 **设置 → 腾讯内网** 的上游连接。

查看 TT Switch 日志：

```bash
tt-switch-cli log --dump --lines 100
```

也可以在网页的 **观测 → 请求记录** 查看失败的模型请求。

### HTTP 429

这是请求限流。等待限流恢复后续跑；需要降低并发时，先停止任务，把 `agibot_local.env` 中的 `AGIBOT_CONCURRENCY` 改为 `2`，重新加载配置再启动。

### `dataset directory is not accessible`

确认原始数据已挂载、当前用户能够读取，且 `AGIBOT_SOURCE` 指向含有 `task_*` 子目录的 AgiBot 数据根目录。

### `semantic configuration changed` 或 `episode selection changed`

当前启动参数与该输出目录记录的配置不同。

继续旧任务时恢复原配置；启动新配置时修改 `AGIBOT_OUTPUT`，使用新目录。两条试跑续跑时同时保留 `--output` 和 `--selection`。

### 同一目录任务锁报错

检查是否已有任务在运行，继续观察已有任务，或停止已有任务后再启动。

## 其他可用命令

只准备源码和数据清单：

```bash
bash agibot.sh prepare
```

`prepare` 会建立输出目录和冻结源码，不调用 GLM。正常 `start` 会自动执行准备流程。

当前代码还提供可选的 `--object-grounding` 实验：在文字标注前增加一次按图识别物体的请求。它会增加 API 调用量，改变本次配置，使用时另建输出目录，并在之后续跑时保留该参数：

```bash
bash agibot.sh start \
  --output "${AGIBOT_OUTPUT}_objects" \
  --object-grounding
```

前面的常规流程已经允许有图像依据的物体和接触描述；这个实验参数增加独立的物体识别步骤。

查看命令帮助：

```bash
bash agibot.sh --help
tt-switch-cli help
```

## 下次启动：已经完成首次配置后

**【服务器终端】**

加载安装路径和原来的任务配置：

```bash
cd /mnt/lyn/workspace/atomic_episode_segmentation
source /jizhicfs/peterrao/miniconda3/bin/activate
export PATH="$PWD/tt-switch-cli/bin:$PATH"
source agibot_local.env

tt-switch-cli start
tt-switch-cli status
bash agibot.sh check
```

确认同一输出目录当前没有任务运行后，前台启动或续跑：

```bash
bash agibot.sh start
```

长时间任务使用第 9.2 步的后台命令；继续两条试跑使用第 8.2 步的完整命令。

这份说明依据当前 [AgiBot 启动脚本](agibot.sh)、[命令入口](scripts/agibot.py)、[AgiBot 详细说明](docs/AGIBOT.md)、[本地 TT Switch 安装说明](tt-switch-cli/README.md) 和安装目录附带的 TT Switch 用户手册编写。
