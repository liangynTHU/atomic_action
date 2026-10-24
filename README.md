# Atomic Episode Segmentation

按两个数据方向组织：**RoboTwin** 和 **AgiBot / ABC-130k 预训练分割**。RoboTwin 内包含原子动作、正常 CoT 和 native recovery；预训练按数据集选择入口。

- [OXE / ABC-130k 非 CoT Pipeline](docs/PRETRAIN_NONCOT.md)：数据集专属 prompt、原始动作格式、真实 Qwen 子任务标注和完整 episode 可视化。
- 当前 ABC 配置：[pretrain_noncot.same_actions.json](configs/pretrain_noncot.same_actions.json)，只合并相邻左右动作对一致的片段，保留不同动作的边界；32–64 行为长度偏好。
- 对齐 RoboTwin 学生格式：[pretrain_noncot.robotwin_aligned.json](configs/pretrain_noncot.robotwin_aligned.json)，单段 system、统一历史观测结构、训练行仅 `messages/images`，保留各数据集真实坐标。
- [AgiBot 分割与标注](docs/AGIBOT.md)：`agibot.sh` / `scripts/agibot.py` 提供完整 episode 的恢复、API 标注、导出与审阅。
- [ABC 全量生产化说明](docs/ABC130K_128GPU_START_PROMPT.md)：当前通用入口处理配置中选定的样本，全量流式分片 runner 仍需实现；示例配置不能直接代表 130k 全量。

- [数据 Pipeline 统一入口](/mnt/lyn/workspace/docs/data/README.md)：把目标、输入、输出和机器／卡数描述给执行者或 Codex，按导航选择流程。
- [Pipeline 运行手册](/mnt/lyn/workspace/docs/data/PIPELINES.md)：配置节点和卡数、启动命令、resume、审计和安装。
- [System prompt 合同](/mnt/lyn/workspace/docs/data/SYSTEM_PROMPTS.md)：system/user/assistant、single/multi、thinking/no-thinking、guide/chunks。
- [数据格式](/mnt/lyn/workspace/docs/data_format.md)：正常与 native recovery 的字段、时序、完整合成示例及实验格式差异。
- [正常 CoT 注意事项](/mnt/lyn/workspace/docs/data/robotwin_cot_guidelines.md)。
- [代码结构和整理决策](/mnt/lyn/workspace/docs/data/CODE_LAYOUT.md)：模块职责、公共层、保留功能、旧入口迁移。
- [旧名到新名映射](docs/CODE_REFACTOR_MAP.json)。
- [Recovery 格式详解](/mnt/lyn/workspace/docs/data/RECOVERY_PIPELINE.md)。
- [Recovery CoT 质量注意事项](/mnt/lyn/workspace/docs/data/RECOVERY_COT_QUALITY.md)：still／短片段、五风格、动作末句、左右／guide、发布门槛和旧数据重建。
- [研究里程碑](docs/MILESTONES.md)：历史记录，不作为当前启动命令。

## 快速开始

```bash
python -m pip install -e .
# 开发与测试
python -m pip install -e '.[dev]'
python -m pytest -q
```

复制 configs 中相应示例，填写实际绝对路径和验收条数：

```bash
python scripts/run_pipeline.py --config /path/to/run.json --dry-run
python scripts/run_pipeline.py --config /path/to/run.json
python scripts/run_pipeline.py --config /path/to/run.json --resume
```

| Pipeline | 配置示例 | 核心模块 |
|---|---|---|
| 原子动作划分 | [atomic_actions.example.json](configs/atomic_actions.example.json) | `atomic_seg.atomic_actions` |
| OXE / ABC 预训练非 CoT | [pretrain_noncot.pilot.json](configs/pretrain_noncot.pilot.json) | `atomic_seg.pretrain` |
| 正常 CoT 生成 | [cot_data.example.json](configs/cot_data.example.json) | `atomic_seg.cot_data` |
| 已通过观察与双 critic 的 CoT 发布 | [cot_materialize.example.json](configs/cot_materialize.example.json) | `atomic_seg.cot_data` |
| Native recovery CoT | [recovery_cot_data.example.json](configs/recovery_cot_data.example.json) | `atomic_seg.recovery_cot` |

正常 CoT 入口接受已对齐的 VLM/VLA/manifest。原始 Parquet 请使用手册中的原子动作/通用适配入口。完整 episode recovery 使用单独的 outcome/phase 格式，不能直接拼入四字段 planner 数据。

## 正常 RoboTwin CoT 的统一流程

WJH 的视觉依据标准已合入 `atomic_seg/cot_data`。流程是：

```text
学生当前/历史输入与真实图片
  → 不看当前答案的 observer
  → writer 生成正文
  → grounding critic 与 action critic 分别审核
  → materialize → 正文审计 → 对齐审计 → release
```

**支持图片有依据的物体名称、颜色、相对位置、接触和持物描述。** 把已观察事实、任务目标和下一动作意图区分开；夹爪开合值本身不能证明接触或成功。四阶段都上传学生可见的 3/6 张真实图片。writer 和 action critic 只额外看到固定 primitive/subtask/chunks，observer 和 grounding critic 不看当前答案。

正文 55–150 词，五种风格按全局行号分配，不要求固定关键词、虚构历史或备选动作。程序保留原 student system/user/images 和最终动作 JSON 文本，追加标准动作末句。两个 critic 都通过才能接受；冲突及证据不足保存供复核并阻止发布，不使用模板 fallback。

已删除独立 WJH 重写／清理目录、选中行文本重写工具和 evidence merge 审阅分支。still、动作标签和短同动作的数值准备工具输出待标注的动作 JSON，正文由主链重新生成。旧 annotation 必须重新满足当前合同，不能仅凭 `accepted_round` 发布。

生成的 `.results.jsonl` 即时保存成功结果，`.review.jsonl` 保存失败样本，`.requests.jsonl` 保存原始请求。中断后 `--resume` 复用已通过结果。release 与安装产物包含 `work/cot5_annotations.jsonl`、`work/REQUEST_JOURNALS.json` 和对应请求日志。

## 多卡原则

本项目 Qwen3.8-Flash-Next 在 H20 上使用每节点 TP8，一个 endpoint 对应 8 卡。64/128/192 卡分别对应 8/16/24 个 endpoints。正常 CoT 只有一个 writer；recovery 的数据 rank 与模型 TP rank 是两件事。

每个分配节点运行 `scripts/serve_model_node.sh`，coordinator 只启动一次。服务和数据任务分别管理，配置中不再硬编码某次实验的 IP 或 rank 行区间。完整命令见运行手册。

## 代码与协议边界

```text
atomic_seg/atomic_actions/    原子动作公共 API
atomic_seg/cot_data/          正常视觉 CoT 四阶段生成、输入准备、审计与发布
atomic_seg/pretrain/          ABC/OXE、AgiBot、RealOmni 的数据集适配与非 CoT
atomic_seg/recovery_cot/      Native / manifest-adapter / full-episode recovery
atomic_seg/common/            文件、模型客户端、数值、CoT envelope、锁、渲染
atomic_seg/prompt_contract.py 固定 planner system 与 response-mode 合同
atomic_seg/action_language.py 动作文字与最终动作总结句
```

预训练非 CoT 使用独立 `atomic_seg.pretrain` adapter/package，复用已有状态分段与模型客户端，OXE/Bridge 和 ABC/YAM 分别声明原始状态及动作合同；旧 ABC V29 专项流程不作为入口。Native recovery 已加入统一 still／短同动作清理和五风格发布约束。V28/V29 不再用作函数或业务模块名，历史数据 schema、图片目录和外部格式 driver 中的标识保留以兼容已有数据。

正常 release 包含 VLM、VLA、动作划分、最终 annotation 和审计报告。生成与安装分开；安装默认拒绝覆盖，显式 `--replace` 会留回退副本。测试使用合成数据或只读检查，不会自动启动 GPU 服务或修改正式训练集。
