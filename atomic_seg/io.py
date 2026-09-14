"""Portable dataset, manifest and JSON I/O helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


CAMERA_KEYS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass(frozen=True)
class SuccessEpisode:
    """One curated, fully correct demonstration."""

    episode_index: int
    task: str = ""
    success: bool = True
    source_episode_id: str | None = None
    success_provenance: str = ""

    def __post_init__(self) -> None:
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if self.success is not True:
            raise ValueError(
                "this repository only accepts fully correct episodes"
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    return value


def load_json(path: Path | str) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_json(path: Path | str, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            yield value


def write_jsonl(
    path: Path | str,
    values: Iterable[Mapping[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(
                json.dumps(
                    _jsonable(value),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def dataset_info(root: Path | str) -> dict[str, Any]:
    value = load_json(Path(root) / "meta" / "info.json")
    if not isinstance(value, dict):
        raise ValueError("meta/info.json must contain an object")
    return value


def validate_end_pose_dataset(root: Path | str) -> Path:
    """Require the 16-D xyz+quaternion+gripper state layout."""

    data_root = Path(root)
    info = dataset_info(data_root)
    shape = (
        info.get("features", {})
        .get("observation.state", {})
        .get("shape")
    )
    if list(shape or []) != [16]:
        raise ValueError(
            f"{data_root} stores observation.state shape {shape}; "
            "the segmentation core requires 16-D end-pose state"
        )
    return data_root


def episode_path(
    root: Path | str,
    episode_index: int,
    chunk_size: int = 1000,
) -> Path:
    episode = int(episode_index)
    return (
        Path(root)
        / "data"
        / f"chunk-{episode // chunk_size:03d}"
        / f"episode_{episode:06d}.parquet"
    )


def video_path(
    root: Path | str,
    episode_index: int,
    camera_key: str,
) -> Path:
    if camera_key not in CAMERA_KEYS:
        raise ValueError(
            f"camera_key must be one of {CAMERA_KEYS}, got {camera_key!r}"
        )
    episode = int(episode_index)
    relative = (
        Path(f"chunk-{episode // 1000:03d}")
        / f"observation.images.{camera_key}"
        / f"episode_{episode:06d}.mp4"
    )
    preferred = Path(root) / "videos" / relative
    if preferred.is_file():
        return preferred
    legacy = Path(root) / "videos_old" / relative
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(preferred)


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
) -> dict[str, Any]:
    path = episode_path(root, episode_index)
    if not path.is_file():
        raise FileNotFoundError(path)
    available = pq.read_schema(path).names
    selected = [name for name in columns if name in available]
    table = pq.read_table(path, columns=selected)
    result: dict[str, Any] = {}
    for name in selected:
        result[name] = np.asarray(table.column(name).to_pylist())
    result["path"] = path
    return result


def load_episode_tasks(root: Path | str) -> dict[int, str]:
    """Return the first published task prompt for every episode."""

    path = Path(root) / "meta" / "episodes.jsonl"
    tasks: dict[int, str] = {}
    for record in read_jsonl(path):
        choices = record.get("tasks") or []
        tasks[int(record["episode_index"])] = (
            str(choices[0]) if choices else ""
        )
    return tasks


def load_success_manifest(path: Path | str) -> list[SuccessEpisode]:
    """Load a manifest and reject non-success/recovery entries."""

    value = load_json(path)
    if not isinstance(value, dict) or not isinstance(
        value.get("episodes"),
        list,
    ):
        raise ValueError("manifest must contain an episodes array")
    episodes: list[SuccessEpisode] = []
    seen: set[int] = set()
    for index, item in enumerate(value["episodes"]):
        if not isinstance(item, dict):
            raise ValueError(f"episodes[{index}] must be an object")
        episode = SuccessEpisode(
            episode_index=int(item["episode_index"]),
            task=str(item.get("task", "")).strip(),
            success=item.get("success") is True,
            source_episode_id=(
                str(item["source_episode_id"])
                if item.get("source_episode_id")
                else None
            ),
            success_provenance=str(
                item.get("success_provenance", "")
            ).strip(),
        )
        if episode.episode_index in seen:
            raise ValueError(
                f"duplicate episode_index {episode.episode_index}"
            )
        seen.add(episode.episode_index)
        episodes.append(episode)
    if not episodes:
        raise ValueError("success manifest contains no episodes")
    return episodes


def resolve_episode_tasks(
    root: Path | str,
    episodes: Sequence[SuccessEpisode],
) -> list[SuccessEpisode]:
    tasks = load_episode_tasks(root)
    output: list[SuccessEpisode] = []
    for episode in episodes:
        task = episode.task or tasks.get(episode.episode_index, "")
        if not task:
            raise ValueError(
                f"episode {episode.episode_index} has no task instruction"
            )
        output.append(
            SuccessEpisode(
                episode_index=episode.episode_index,
                task=task,
                success=True,
                source_episode_id=(
                    episode.source_episode_id
                    or f"episode_{episode.episode_index:06d}"
                ),
                success_provenance=episode.success_provenance,
            )
        )
    return output
