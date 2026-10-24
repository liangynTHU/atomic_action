"""Build the normal-success action segmentation manifest in recovery shape."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping

from atomic_seg.common.artifacts import compact_json as _compact_json
from atomic_seg.common.artifacts import iter_jsonl as _source_rows
from atomic_seg.common.artifacts import sha256_file as _sha256
from atomic_seg.cot_data.records import iter_vla as _iter_vla

SCHEMA_VERSION = "robotwin_official_v28_action_manifest_v2"


def _manifest_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "start_frame": int(segment["start_frame"]),
        "end_frame": int(segment["end_frame"]),
        "left_action": str(segment["left_action"]),
        "right_action": str(segment["right_action"]),
        "segment_index": int(segment["source_segment_id"]),
        "primary_action_verb": str(segment["primary_action_verb"]),
        "global_block_index": int(segment["global_block_index"]),
        "source_global_block_index": int(
            segment.get("source_global_block_index", segment["global_block_index"])
        ),
        "sub_task": str(segment["sub_task"]),
        "guide_action": copy.deepcopy(segment["guide_action"]),
        "guide_action_semantics": str(segment["guide_action_semantics"]),
        "num_chunks": int(segment["num_chunks"]),
    }
    if "merged_source_segment_indices" in segment:
        output["merged_source_segment_indices"] = [
            int(item) for item in segment["merged_source_segment_indices"]
        ]
    return output


def build_action_manifest(
    *,
    vla_path: Path,
    source_manifest_path: Path,
    output_path: Path,
    report_path: Path,
    code_version: str = "robotwin_cot_current",
    confirmed_corrections: int | None = None,
    segmentation_policy: str = (
        "nonterminal both-still absorbed; one >=8-row terminal both-still stage allowed"
    ),
) -> dict[str, Any]:
    """Write one final-boundary episode row matching recovery's manifest role."""

    source_by_episode = {
        int(row["global_episode_index"]): row
        for row in _source_rows(source_manifest_path)
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    episodes = 0
    segments = 0
    previous_global = -1
    with temporary.open("w", encoding="utf-8") as destination:
        for _training_key, episode in _iter_vla(vla_path):
            global_index = int(episode["global_episode_index"])
            if global_index <= previous_global:
                raise ValueError("VLA episodes are not globally ordered")
            source = source_by_episode.get(global_index)
            if source is None:
                raise ValueError(f"missing source manifest episode {global_index}")
            episode_segments = [
                _manifest_segment(segment) for segment in episode["segments"]
            ]
            if not episode_segments:
                raise ValueError(f"episode {global_index} has no segments")
            if episode_segments[0]["start_frame"] != 0:
                raise ValueError(f"episode {global_index} does not start at zero")
            if episode_segments[-1]["end_frame"] != int(episode["all_frames"]) - 1:
                raise ValueError(f"episode {global_index} final frame mismatch")
            for index, segment in enumerate(episode_segments):
                if segment["segment_index"] != index:
                    raise ValueError("manifest segment indices are not contiguous")
                if (
                    index
                    and segment["start_frame"]
                    != episode_segments[index - 1]["end_frame"] + 1
                ):
                    raise ValueError("manifest frame ranges are not contiguous")
            value = {
                "schema_version": SCHEMA_VERSION,
                "code_version": code_version,
                "global_episode_index": global_index,
                "training_episode_index": int(episode["training_episode_index"]),
                "task_index": int(episode["task_index"]),
                "episode_id": str(episode["source_episode_id"]),
                "task_name": str(episode["task_name"]),
                "source_variant": str(episode["source_variant"]),
                "official_episode_index": int(episode["official_episode_index"]),
                "task_instruction": str(episode["task_instruction"]),
                "all_frames": int(episode["all_frames"]),
                "training_eligible": True,
                "segment_count": len(episode_segments),
                "segments": episode_segments,
                "source_paths": copy.deepcopy(source.get("source_paths", {})),
                "global_block_start": int(episode_segments[0]["global_block_index"]),
                "segmentation_refinement": {
                    "policy": segmentation_policy,
                    "action_language": "actuator_grounded",
                    "confirmed_raw_trajectory_label_corrections": confirmed_corrections,
                },
            }
            destination.write(_compact_json(value) + "\n")
            episodes += 1
            segments += len(episode_segments)
            previous_global = global_index
    os.replace(temporary, output_path)
    report = {
        "schema_version": "robotwin_official_v28_action_manifest_report_v1",
        "episodes": episodes,
        "segments": segments,
        "vla_path": str(vla_path),
        "source_identity_manifest": str(source_manifest_path),
        "output": str(output_path),
        "published_action_segmentation": "data/action_segmentation/train.jsonl",
        "metadata_mirror": "metadata/source_manifest.jsonl",
        "vla_sha256": _sha256(vla_path),
        "output_sha256": _sha256(output_path),
        "recovery_compatible_role": (
            "one episode row with segment_count and a complete segments array"
        ),
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
