# Atomic Episode Segmentation

这是一个面向**完全正确机器人演示**的精简数据流水线。仓库只保留：

1. 从 16 维双臂末端状态和时间戳划分 atomic action；
2. 生成不带 CoT 的纯动作监督；
3. 为完整成功 episode 批量生成多模态 CoT 数据；
4. 为单条成功 episode 生成 CoT 数据；
5. 生成分割可视化和完整 episode 可视化。

失败、漂移、策略偏差和 recovery 数据不在当前运行范围内。历史研究过程统一记录在
[`docs/MILESTONES.md`](docs/MILESTONES.md)，但不再作为生产代码入口。

## 1. 最终目录

```text
atomic_seg/
  config.py          分割参数
  features.py        状态特征
  geometry.py        四元数运算
  segmentation.py    最终 state-only 双臂分割
  io.py              数据集、manifest、JSON/JSONL I/O
  media.py           三相机视频抽帧
  generation.py      action-only 与 CoT 数据生成
  visualization.py   两类可视化

scripts/
  segment_episodes.py
  generate_action_data.py
  generate_cot_data.py
  generate_episode_cot.py
  build_visualizations.py

schemas/
  action_dataset.schema.json
  cot_dataset.schema.json

examples/
  success_episodes.json
  visualizations/

docs/
  MILESTONES.md
```

运行代码、脚本、schema 和测试均使用稳定功能名称，不使用研究迭代编号。

## 2. 输入数据合同

当前分割算法要求 RoboTwin/LeRobot 风格目录：

```text
DATA_ROOT/
  meta/info.json
  meta/episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.cam_high/episode_000000.mp4
  videos/chunk-000/observation.images.cam_left_wrist/episode_000000.mp4
  videos/chunk-000/observation.images.cam_right_wrist/episode_000000.mp4
```

Parquet 至少包含：

```text
observation.state   [T, 16]
action              [T, 16]
timestamp           [T]
frame_index         [T]
```

16 维顺序固定为：

```text
left xyz + left quaternion(wxyz) + left gripper
right xyz + right quaternion(wxyz) + right gripper
```

边界判断只使用 `observation.state` 和 `timestamp`。图像、任务文本、action 和
success 标签不会移动边界。

## 3. 成功 episode manifest

所有生成入口都要求显式的成功清单：

```json
{
  "schema": "successful_episode_manifest",
  "episodes": [
    {
      "episode_index": 18,
      "success": true,
      "success_provenance": "Manually verified successful demonstration."
    }
  ]
}
```

规则：

- `success` 必须严格为 `true`；
- `success=false` 会立即报错；
- 不接受 recovery、failure prefix 或结果不明确的轨迹；
- `task` 可选，缺失时读取 `meta/episodes.jsonl` 的第一条任务文本。

## 4. 安装

```bash
python -m pip install -r requirements.txt
```

或者：

```bash
python -m pip install -e .
```

## 5. Atomic action 分割

```bash
python scripts/segment_episodes.py \
  --data-root /path/to/robotwin-endpose \
  --manifest /path/to/success_episodes.json \
  --output-dir artifacts/segmentation
```

输出每条 episode 的：

- joint boundaries；
- left/right 独立 boundaries；
- boundary evidence；
- 左右臂 primitive timeline；
- 每段 `left_action` / `right_action`；
- coordination relation。

核心 Python API：

```python
from atomic_seg import segment_episode

result = segment_episode(states, timestamps)
```

算法包含：

- timestamp 归一化的物理速度；
- motion-energy pause evidence；
- direction-neutral gripper transition evidence；
- persistent local phase；
- successor-trend-onset refinement；
- 左右臂独立 primitive timeline；
- 派生的双臂 coordination timeline。

## 6. 生成不带 CoT 的动作数据

```bash
python scripts/generate_action_data.py \
  --data-root /path/to/robotwin-endpose \
  --manifest /path/to/success_episodes.json \
  --output-dir artifacts/action_data
```

主要输出：

```text
artifacts/action_data/train.json
artifacts/action_data/manifest.json
```

每个 segment 包含：

```text
start_frame / end_frame
left_action / right_action
primary_action_verb
sub_task
guide_action
num_chunks
```

数值目标定义为：

```text
guide_action = action[end_frame] - observation.state[start_frame]
num_chunks   = ceil((end_frame - start_frame + 1) / 8)
```

纯动作数据禁止出现 `cot`、`reasoning`、`chain_of_thought` 或
`assistant_response`。

## 7. 批量生成 CoT 数据

### 7.1 本地结构验证

`template` 模式用于测试格式、抽帧、数据对齐和可视化，不应被当作高质量模型标注：

```bash
python scripts/generate_cot_data.py \
  --data-root /path/to/robotwin-endpose \
  --manifest /path/to/success_episodes.json \
  --output-dir artifacts/cot_data \
  --reasoner template
```

### 7.2 OpenAI-compatible 多模态模型

```bash
python scripts/generate_cot_data.py \
  --data-root /path/to/robotwin-endpose \
  --manifest /path/to/success_manifest.json \
  --output-dir artifacts/cot_data \
  --reasoner openai-compatible \
  --endpoint http://127.0.0.1:8007 \
  --model Qwen/Qwen3.8-Flash-Next
```

如服务需要 API key，只设置环境变量，不要写入仓库：

```bash
export OPENAI_API_KEY='...'
```

主要输出：

```text
artifacts/cot_data/action_train.json
artifacts/cot_data/cot_train.jsonl
artifacts/cot_data/assets/
artifacts/cot_data/manifest.json
```

每个 atomic segment 对应一条 CoT row：

```text
当前三相机图像
+ 可选的上一决策点三相机图像
+ 当前/历史 robot state
+ 已完成 action history
→ <think>pre-action reasoning</think>
→ {guide_action, primary_action_verb, sub_task, num_chunks}
```

最终 human turn 不包含当前 `guide_action` 或 `sub_task`；当前动作只出现在
assistant 输出和审计用 `target` 元数据中。

## 8. 单条 episode 生成 CoT

```bash
python scripts/generate_episode_cot.py \
  --data-root /path/to/robotwin-endpose \
  --episode 18 \
  --output-dir artifacts/episode_18 \
  --reasoner openai-compatible \
  --endpoint http://127.0.0.1:8007 \
  --model Qwen/Qwen3.8-Flash-Next
```

若 `meta/episodes.jsonl` 没有合适任务文本，可额外传入：

```bash
--task "Hold the Coca-Cola bottle upright after lifting with the right arm"
```

该入口和批量生成器共用完全相同的实现和数据合同。

## 9. 两类可视化

```bash
python scripts/build_visualizations.py \
  --data-root /path/to/robotwin-endpose \
  --manifest /path/to/three_successful_episodes.json \
  --output-dir artifacts/visualizations \
  --zip-path artifacts/atomic_episode_visualizations.zip
```

`examples/success_episodes.example.json` 是 manifest 模板。请复制到本地、替换为
经过人工确认的三条成功 episode；脚本要求恰好三条，随后为每条生成两类可视化：

1. `artifacts/visualizations/segmentation/episode_XXXXXX.svg`
   - 双臂能量；
   - left/right boundary；
   - joint coordination boundary；
   - atomic segment table。
2. `artifacts/visualizations/episodes/episode_XXXXXX.html`
   - 三路完整视频；
   - 每段三相机当前关键帧；
   - action-only supervision；
   - 对应的嵌入式 CoT。

总入口：

```text
artifacts/visualizations/index.html
```

可直接下载或传输的完整包：

```text
artifacts/atomic_episode_visualizations.zip
```

ZIP 包含 3 个分割 SVG、3 个完整 episode HTML、逐段关键帧和 9 路完整相机视频。
由于这些内容来自本地数据集，`artifacts/`、`visualization_archives/` 和媒体 ZIP
均被 Git 忽略，不上传到代码仓库。

## 10. 测试

```bash
python -m pytest -q
```

测试覆盖：

- 时间归一化和 successor onset；
- 双臂 boundary fusion；
- success-only 拒绝规则；
- action-only 无 CoT；
- CoT target isolation；
- 使用纯合成轨迹/图片/占位视频生成三个 SVG 和三个完整 episode HTML；
- 合成 HTML 的所有本地引用存在，合成 ZIP 的 3 SVG、3 HTML、9 MP4 合同成立。

## 11. 当前范围

当前仓库只处理**完全正确的完整 episode**：

- 不推断 failure cause；
- 不处理 drift；
- 不拼接 policy prefix 与 recovery rollout；
- 不生成 recovery reasoning；
- 不把历史 failure/recovery 代码留在生产路径。

如未来重新开展 failure/recovery 项目，应建立独立仓库或独立 package，而不是再次
混入这里的成功演示流水线。
