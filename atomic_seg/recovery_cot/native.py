"""Native recovery trajectories, semantic boundaries and source manifest."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from atomic_seg.common.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    bounded_parallel,
    compact_json,
    jsonl_rows,
    utc_now,
)
from atomic_seg.config import SegmentationConfig
from atomic_seg.features import segment_motion_summary
from atomic_seg.segmentation import segment_episode, segments_from_boundaries

from .contracts import CODE_VERSION, MANIFEST_SCHEMA, STATE_LAYOUT, VALID_PRIMITIVES
from .policy import QUALITY_POLICY, normalize_segments, validate_segments


def manifest_paths(output_root: Path) -> tuple[Path, Path]:
    return (
        output_root / "metadata/source_manifest.jsonl",
        output_root / "metadata/MANIFEST_SUMMARY.json",
    )


def _native_paths(record: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    job_root = Path(str(record["output_dir"]))
    recovery = sorted(job_root.glob("recovery_rollouts/*/*/*/episode.json"))
    source = sorted(job_root.glob("raw_rollouts/*/*/episode.json"))
    if len(recovery) != 1 or len(source) != 1:
        raise ValueError(
            f"{record.get('job_id')}: expected one source and one recovery episode; "
            f"found source={len(source)} recovery={len(recovery)}"
        )
    hdf5 = recovery[0].with_name("episode.hdf5")
    if not hdf5.is_file():
        raise FileNotFoundError(hdf5)
    return source[0], recovery[0], hdf5


def _endpose_vector(group: h5py.Group) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(group["left_endpose"], dtype=np.float64),
            np.asarray(group["left_gripper"], dtype=np.float64),
            np.asarray(group["right_endpose"], dtype=np.float64),
            np.asarray(group["right_gripper"], dtype=np.float64),
        ],
        axis=-1,
    )


def load_native_endpose(
    hdf5_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    with h5py.File(hdf5_path, "r") as source:
        states = _endpose_vector(source["endpose"])
        final_state = _endpose_vector(
            source["final_observation/robotwin_observation/endpose"]
        )
        if final_state.shape != (16,):
            raise ValueError(f"{hdf5_path}: final endpose must be 16-D")
        actions = np.vstack([states[1:], final_state[None, :]])
        timestamps = np.asarray(source["transition/timestamp"], dtype=np.float64)
        extras = [
            json.loads(value) if value else {}
            for value in source["transition/extra_json"].asstr()[:]
        ]
    if states.ndim != 2 or states.shape[1] != 16:
        raise ValueError(f"{hdf5_path}: endpose state is not [T,16]")
    if actions.shape != states.shape or len(timestamps) != len(states):
        raise ValueError(f"{hdf5_path}: state/action/timestamp length mismatch")
    if len(extras) != len(states):
        raise ValueError(f"{hdf5_path}: transition metadata length mismatch")
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
        raise ValueError(f"{hdf5_path}: non-finite endpose values")
    return states, actions, timestamps, extras


def _primitive_for_arm(
    states: np.ndarray,
    actions: np.ndarray,
    start: int,
    end: int,
    arm: str,
    config: SegmentationConfig,
) -> str:
    trajectory = np.vstack([states[start : end + 1], actions[end][None, :]])
    summary = segment_motion_summary(
        trajectory,
        0,
        len(trajectory) - 1,
        arm,
    )
    gripper_delta = float(summary["gripper_delta"])
    if abs(gripper_delta) >= max(0.08, config.gripper_change_threshold * 4):
        return "close_gripper" if gripper_delta < 0.0 else "open_gripper"
    displacement = np.asarray(summary["displacement"], dtype=np.float64)
    vertical = np.asarray(config.vertical_direction, dtype=np.float64)
    vertical /= max(float(np.linalg.norm(vertical)), 1e-12)
    vertical_displacement = float(np.dot(displacement, vertical))
    horizontal = float(np.linalg.norm(displacement - vertical_displacement * vertical))
    translation = float(summary["translation_norm"])
    rotation = float(summary["rotation_path"])
    # Net displacement alone mislabels out-and-back motion as stationary.
    gripper_path = np.ptp(trajectory[:, 7 if arm == "left" else 15])
    if (
        float(summary["path_length"]) < 0.005
        and rotation < 0.03
        and gripper_path < 0.08
    ):
        return "still"
    if (
        abs(vertical_displacement) > max(0.015, 0.60 * translation)
        and abs(vertical_displacement) > horizontal
    ):
        return "lift" if vertical_displacement > 0.0 else "lower"
    if rotation / 0.10 > 4.5 * max(translation / 0.08, 1e-6):
        return "turn"
    return "move"


def _segment_group_index(
    start: int,
    group_starts: Sequence[int],
) -> int:
    result = 0
    for index, value in enumerate(group_starts):
        if value > start:
            break
        result = index
    return result


def recovery_segments(
    states: np.ndarray,
    actions: np.ndarray,
    timestamps: np.ndarray,
    extras: Sequence[Mapping[str, Any]],
    *,
    config: SegmentationConfig | None = None,
) -> list[dict[str, Any]]:
    """Combine state-only atomic boundaries with native decision boundaries."""

    selected = config or SegmentationConfig()
    state_result = segment_episode(states, timestamps, selected)
    native_starts = sorted(
        {
            0,
            *(
                index
                for index, value in enumerate(extras)
                if index > 0 and value.get("new_decision_boundary") is True
            ),
        }
    )
    boundaries = sorted(
        set(int(value) for value in state_result["boundaries"]) | set(native_starts[1:])
    )
    raw = segments_from_boundaries(boundaries, len(states))
    labelled: list[dict[str, Any]] = []
    for segment in raw:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        group_index = _segment_group_index(start, native_starts)
        left = _primitive_for_arm(states, actions, start, end, "left", selected)
        right = _primitive_for_arm(states, actions, start, end, "right", selected)
        labelled.append(
            {
                "start_frame": start,
                "end_frame": end,
                "left_action": left,
                "right_action": right,
                "native_group_index": group_index,
                "native_semantic_action": str(
                    extras[native_starts[group_index]].get(
                        "semantic_action", "recovery_action"
                    )
                ),
                "native_event_stage": str(
                    extras[native_starts[group_index]].get("event_stage", "recovery")
                ),
            }
        )

    merged = normalize_segments(labelled)
    # Merging changes the action span. Recompute labels from actual motion,
    # never keep a label merely because it belonged to the chosen owner.
    for _ in range(len(labelled) + 1):
        before = [
            (s["start_frame"], s["end_frame"], s["left_action"], s["right_action"])
            for s in merged
        ]
        for step in merged:
            for arm in ("left", "right"):
                step[f"{arm}_action"] = _primitive_for_arm(
                    states,
                    actions,
                    int(step["start_frame"]),
                    int(step["end_frame"]),
                    arm,
                    selected,
                )
        merged = normalize_segments(merged)
        after = [
            (s["start_frame"], s["end_frame"], s["left_action"], s["right_action"])
            for s in merged
        ]
        if before == after:
            break
    else:
        raise ValueError("recovery action normalization did not converge")

    for index, item in enumerate(merged):
        item["segment_index"] = index
        item["primary_action_verb"] = (
            f"left:{item['left_action']}, right:{item['right_action']}"
        )
        if item["left_action"] not in VALID_PRIMITIVES:
            raise ValueError(f"unsupported left primitive: {item['left_action']}")
        if item["right_action"] not in VALID_PRIMITIVES:
            raise ValueError(f"unsupported right primitive: {item['right_action']}")
    if not merged or int(merged[0]["start_frame"]) != 0:
        raise ValueError("recovery segmentation must begin at frame zero")
    if int(merged[-1]["end_frame"]) != len(states) - 1:
        raise ValueError("recovery segmentation must cover the final action row")
    for previous, current in zip(merged, merged[1:]):
        if int(current["start_frame"]) != int(previous["end_frame"]) + 1:
            raise ValueError("recovery segmentation is not contiguous")
    return merged


def _generic_subtask(left: str, right: str) -> str:
    clauses: list[str] = []
    for arm, action in (("left", left), ("right", right)):
        if action == "still":
            clauses.append(f"keep the {arm} arm still")
        elif action == "move":
            clauses.append(f"move the {arm} arm toward the recovery target")
        elif action == "turn":
            clauses.append(f"turn the {arm} end effector toward the recovery target")
        elif action == "lift":
            clauses.append(f"lift the {arm} arm")
        elif action == "lower":
            clauses.append(f"lower the {arm} arm")
        elif action == "close_gripper":
            clauses.append(f"close the {arm} gripper")
        elif action == "open_gripper":
            clauses.append(f"open the {arm} gripper")
        else:
            raise ValueError(f"unsupported primitive {action!r}")
    return "robot " + " while ".join(clauses)


def build_manifest_entry(
    item: tuple[int, Mapping[str, Any]],
) -> dict[str, Any]:
    index, audit_record = item
    source_json_path, recovery_json_path, hdf5_path = _native_paths(audit_record)
    source_json = json.loads(source_json_path.read_text(encoding="utf-8"))
    recovery_json = json.loads(recovery_json_path.read_text(encoding="utf-8"))
    if (
        audit_record.get("valid") is not True
        or audit_record.get("pair_audit_ok") is not True
    ):
        raise ValueError(f"{audit_record.get('job_id')}: production audit is not valid")
    required = {
        "eligible_for_training": True,
        "pair_audit_status": "passed",
        "source_failure_verified": True,
        "source_failure_mode": "execution_failure",
        "recovery_success": True,
        "new_decision_boundary": True,
        "direct_continuation": False,
    }
    for key, expected in required.items():
        if recovery_json.get(key) != expected:
            raise ValueError(
                f"{audit_record.get('job_id')}: {key}={recovery_json.get(key)!r}, "
                f"expected {expected!r}"
            )
    states, actions, timestamps, extras = load_native_endpose(hdf5_path)
    segments = recovery_segments(states, actions, timestamps, extras)
    task = str(audit_record["instruction"]).strip()
    if task != str(recovery_json.get("instruction", "")).strip():
        raise ValueError(f"{audit_record.get('job_id')}: instruction mismatch")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "code_version": CODE_VERSION,
        "global_episode_index": index,
        "training_episode_index": index,
        "task_index": index,
        "episode_id": str(recovery_json["episode_id"]),
        "job_id": str(audit_record["job_id"]),
        "task_name": str(audit_record["task"]),
        "source_variant": "Recovery",
        "official_episode_index": index,
        "task_instruction": task,
        "recovery_level": int(recovery_json["recovery_level"]),
        "recovery_level_name": str(recovery_json.get("recovery_level_name", "")),
        "failure_type": str(recovery_json.get("failure_type", "unknown")),
        "source_failure_type": str(source_json.get("failure_type", "unknown")),
        "all_frames": int(len(states)),
        "training_eligible": True,
        "segment_count": len(segments),
        "segments": segments,
        "source_paths": {
            "source_episode_json": str(source_json_path),
            "recovery_episode_json": str(recovery_json_path),
            "recovery_hdf5": str(hdf5_path),
            "pair_audit": str(
                Path(str(audit_record["output_dir"])) / "pair_audit.json"
            ),
        },
    }


def create_manifest(
    source_root: Path,
    output_root: Path,
    *,
    workers: int = 16,
    limit_episodes: int | None = None,
) -> dict[str, Any]:
    if any(path.exists() for path in manifest_paths(output_root)):
        raise FileExistsError("manifest already exists; use a fresh output root")
    audit_path = source_root / "_state/production_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    raw_records = audit.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("source production audit requires nonempty records")
    if audit.get("complete") is not True or int(
        audit.get("completed_markers", 0)
    ) != len(raw_records):
        raise ValueError("source production audit markers must cover all records")
    if limit_episodes is not None:
        raw_records = raw_records[: int(limit_episodes)]
    indexed = list(enumerate(raw_records))
    rows_by_index: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    for item, result, error in bounded_parallel(
        indexed,
        workers=workers,
        fn=build_manifest_entry,
    ):
        index, source = item
        if error is not None:
            failures.append(
                {
                    "global_episode_index": index,
                    "job_id": source.get("job_id"),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        assert result is not None
        rows_by_index[index] = result
        if len(rows_by_index) == 1 or len(rows_by_index) % 250 == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[manifest] {len(rows_by_index)}/{len(indexed)} "
                f"episodes ({len(rows_by_index) / elapsed:.2f}/s)",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"manifest preparation failed for {len(failures)} episodes: "
            + compact_json(failures[:10])
        )
    rows = [rows_by_index[index] for index in range(len(indexed))]
    block_index = 0
    for row in rows:
        row["global_block_start"] = block_index
        for segment in row["segments"]:
            segment["global_block_index"] = block_index
            block_index += 1
    manifest_path, summary_path = manifest_paths(output_root)
    summary = {
        "schema_version": "robotwin_recovery_v28_manifest_summary_v1",
        "code_version": CODE_VERSION,
        "created_at_utc": utc_now(),
        "source_root": str(source_root.resolve()),
        "source_audit": str(audit_path.resolve()),
        "episodes": len(rows),
        "blocks": block_index,
        "state_layout": list(STATE_LAYOUT),
        "quality_policy": QUALITY_POLICY,
        "segmentation_policy": (
            "union of 16-D state-only atomic boundaries and audited native "
            "recovery decision boundaries; absorb initial/middle both-still, "
            "merge short identical-action neighbors, retain distinct primitive "
            "transitions and record original spans as provenance"
        ),
        "recommended_outputs": {
            "vlm": str(output_root / "data/vlm/train.jsonl"),
            "vla": str(output_root / "data/vla/train.json"),
        },
    }
    atomic_write_jsonl(manifest_path, rows)
    atomic_write_json(summary_path, summary)
    return summary


def load_manifest(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path, summary_path = manifest_paths(output_root)
    rows = jsonl_rows(manifest_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("code_version") != CODE_VERSION:
        raise ValueError("manifest code version mismatch")
    if summary.get("quality_policy") != QUALITY_POLICY:
        raise ValueError("old recovery manifest: regenerate in a fresh output root")
    if len(rows) != int(summary["episodes"]):
        raise ValueError("manifest episode count mismatch")
    next_block = 0
    for index, row in enumerate(rows):
        if (
            int(row["global_episode_index"]) != index
            or int(row["global_block_start"]) != next_block
        ):
            raise ValueError("manifest episode/block ordering mismatch")
        if int(row["segment_count"]) != len(row["segments"]):
            raise ValueError("manifest segment count mismatch")
        validate_segments(row["segments"])
        for segment in row["segments"]:
            if int(segment["global_block_index"]) != next_block:
                raise ValueError("manifest global block IDs are not contiguous")
            next_block += 1
        if int(row["segments"][-1]["end_frame"]) + 1 != int(row["all_frames"]):
            raise ValueError("manifest does not cover all native action rows")
    if next_block != int(summary["blocks"]):
        raise ValueError("manifest block total mismatch")
    return rows, summary
