"""Dataset and JSON I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pyarrow.parquet as pq


DEFAULT_JOINT_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/"
    "fastwam/data/robotwin2.0"
)
DEFAULT_ENDPOSE_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/"
    "cosmos3/data/robotwin2.0-endpose"
)
DEFAULT_REFERENCE_JSON = Path(
    "/mnt/ybw/workspace/divide_action/outputs/full/"
    "atomic_segments_cot_merged_grasp_task01.json"
)


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path | str, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)


def load_info(root: Path | str) -> Dict[str, Any]:
    return load_json(Path(root) / "meta" / "info.json")


def load_episode_tasks(root: Path | str) -> Dict[int, str]:
    path = Path(root) / "meta" / "episodes.jsonl"
    tasks: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            choices = record.get("tasks") or []
            tasks[int(record["episode_index"])] = choices[0] if choices else ""
    return tasks


def episode_path(root: Path | str, episode_index: int, chunk_size: int = 1000) -> Path:
    return (
        Path(root)
        / "data"
        / f"chunk-{episode_index // chunk_size:03d}"
        / f"episode_{episode_index:06d}.parquet"
    )


def load_episode(
    root: Path | str,
    episode_index: int,
    columns: Sequence[str] = (
        "observation.state",
        "action",
        "timestamp",
        "frame_index",
        "task_index",
    ),
) -> Dict[str, np.ndarray]:
    path = episode_path(root, episode_index)
    available = pq.read_schema(path).names
    selected = [name for name in columns if name in available]
    table = pq.read_table(path, columns=selected)
    result: Dict[str, np.ndarray] = {}
    for name in selected:
        values = table.column(name).to_pylist()
        result[name] = np.asarray(values)
    result["path"] = np.asarray(str(path))
    return result


def infer_pose_root(requested_root: Path | str) -> Path:
    """Resolve a 16D xyz+quaternion+gripper dataset.

    The path supplied by the user is the 14D joint-space dataset. The same
    episodes have already been converted to end-effector pose under the
    sibling cosmos3 tree. We prefer an explicitly supplied 16D root and fall
    back to that known paired dataset otherwise.
    """

    requested = Path(requested_root)
    info = load_info(requested)
    shape = info["features"]["observation.state"]["shape"]
    if list(shape) == [16]:
        return requested
    if DEFAULT_ENDPOSE_ROOT.exists():
        return DEFAULT_ENDPOSE_ROOT
    raise ValueError(
        f"{requested} stores shape {shape}, not 16D end-pose data, and the paired "
        f"end-pose root {DEFAULT_ENDPOSE_ROOT} is unavailable."
    )


def load_reference(
    path: Path | str | None, episode_index: int
) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    source = Path(path)
    if not source.exists():
        return None
    blob = load_json(source)
    record = blob.get(str(episode_index)) if isinstance(blob, dict) else None
    return record


def parse_episode_ids(spec: str | Iterable[int]) -> List[int]:
    if isinstance(spec, str):
        return [int(token) for token in spec.split(",") if token.strip()]
    return [int(value) for value in spec]
