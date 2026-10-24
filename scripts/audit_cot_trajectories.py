#!/usr/bin/env python3
"""Read-only extended semantic and diversity audit for RoboTwin."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from atomic_seg.action_language import canonical_sub_task, cot_final_action_sentence
from atomic_seg.common.artifacts import iter_jsonl as rows
from atomic_seg.cot_data.records import iter_vla as _iter_vla

THINK_RE = re.compile(r"^<think>\n(.*)\n</think>\n\n(\{.*\})$", re.DOTALL)
WORD_RE = re.compile(r"[a-z0-9']+")


def actions(value: str) -> dict[str, str]:
    result = {}
    for item in value.split(","):
        arm, separator, action = item.strip().partition(":")
        if not separator:
            raise ValueError(f"bad action pair: {value}")
        result[arm] = action
    if set(result) != {"left", "right"}:
        raise ValueError(f"bad action pair: {value}")
    return result


def quaternion_angle(first: Sequence[float], second: Sequence[float]) -> float:
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return math.nan
    dot = abs(float(np.dot(a, b)) / denominator)
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def endpoint_metrics(row: Mapping[str, Any], offset: int) -> dict[str, float]:
    current = np.asarray(row["current_state"], dtype=np.float64)
    target = np.asarray(row["guide_action_target_state"], dtype=np.float64)
    delta = target[offset : offset + 3] - current[offset : offset + 3]
    return {
        "translation_m": float(np.linalg.norm(delta)),
        "horizontal_m": float(np.linalg.norm(delta[:2])),
        "vertical_m": float(delta[2]),
        "rotation_rad": quaternion_angle(
            current[offset + 3 : offset + 7], target[offset + 3 : offset + 7]
        ),
        "gripper_delta": float(target[offset + 7] - current[offset + 7]),
    }


def normalize_opening(value: str) -> str:
    value = value.lower()
    replacements = (
        (r"\b(left|right)\b", "arm"),
        (r"\b(open|closed|close|opening|closing)\b", "gripper_state"),
        (
            r"\b(move|moves|moving|moved|reposition|repositions|repositioning|"
            r"shift|shifts|shifting|translate|translates|translating)\b",
            "motion",
        ),
        (r"\b(turn|turns|turning|rotate|rotates|rotating|rotation)\b", "rotation"),
        (r"\b(lift|lifts|lifting|raise|raises|raising|upward)\b", "vertical"),
        (
            r"\b(lower|lowers|lowering|descend|descends|descending|downward)\b",
            "vertical",
        ),
        (r"\b\d+(?:\.\d+)?\b", "number"),
    )
    for pattern, replacement in replacements:
        value = re.sub(pattern, replacement, value)
    return " ".join(WORD_RE.findall(value))


def append_example(
    store: dict[str, list[dict[str, Any]]], key: str, value: dict[str, Any]
) -> None:
    if len(store[key]) < 20:
        store[key].append(value)


def structural_audit(
    root: Path, annotations: Path, out: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    vlm_path = root / "data/vlm/train.jsonl"
    vla_path = root / "data/vla/train.json"
    manifest_path = root / "metadata/source_manifest.jsonl"
    vlm = rows(vlm_path)
    manifest = rows(manifest_path)
    annotation = rows(annotations)
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    length_counts: Counter[int] = Counter()
    short_action_counts: Counter[str] = Counter()
    action_pair_counts: Counter[str] = Counter()
    adjacent_same_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    first_sentence_counts: Counter[str] = Counter()
    normalized_first_sentence_counts: Counter[str] = Counter()
    prefix8_counts: Counter[str] = Counter()
    sentence_counts: Counter[int] = Counter()
    style_openings: dict[str, Counter[str]] = defaultdict(Counter)
    exact_reasoning_hashes: set[str] = set()
    task_sequences: dict[str, Counter[str]] = defaultdict(Counter)
    raw_jobs: list[dict[str, Any]] = []
    row_index = 0

    for training_key, episode in _iter_vla(vla_path):
        counts["episodes"] += 1
        try:
            manifest_episode = next(manifest)
        except StopIteration as exc:
            raise ValueError("manifest ended before VLA") from exc
        if int(training_key) != int(episode["training_episode_index"]):
            issues["vla_training_index_mismatch"] += 1
        episode_fields = (
            "global_episode_index",
            "training_episode_index",
            "task_index",
            "task_name",
            "source_variant",
            "official_episode_index",
        )
        for key in episode_fields:
            if episode.get(key) != manifest_episode.get(key):
                issues[f"manifest_vla_episode_{key}_mismatch"] += 1
        if episode.get("task") != manifest_episode.get("task_instruction"):
            issues["manifest_vla_task_mismatch"] += 1
        segments = episode["segments"]
        manifest_segments = manifest_episode["segments"]
        if len(segments) != len(manifest_segments):
            issues["manifest_vla_segment_count_mismatch"] += 1
        sequence: list[str] = []
        raw_segments: list[dict[str, Any]] = []
        previous_pair: str | None = None
        previous_target: Mapping[str, Any] | None = None
        previous_end: int | None = None
        for segment_index, (segment, manifest_segment) in enumerate(
            zip(segments, manifest_segments)
        ):
            try:
                row = next(vlm)
                ann = next(annotation)
            except StopIteration as exc:
                raise ValueError("VLM or annotation ended before VLA") from exc
            counts["segments"] += 1
            if int(row.get("episode_index", -1)) != int(
                episode["global_episode_index"]
            ):
                issues["vlm_vla_episode_mismatch"] += 1
            if int(row.get("segment_index", -1)) != segment_index:
                issues["vlm_vla_segment_index_mismatch"] += 1
            if int(ann.get("row_index", -1)) != row_index:
                issues["annotation_row_index_mismatch"] += 1
            if (
                int(ann.get("episode_index", -1)),
                int(ann.get("segment_index", -1)),
            ) != (int(row.get("episode_index", -2)), int(row.get("segment_index", -2))):
                issues["annotation_vlm_key_mismatch"] += 1
            if row.get("task") != episode.get("task"):
                issues["vlm_vla_task_mismatch"] += 1
            target = row["target"]
            pair = str(target["primary_action_verb"])
            action = actions(pair)
            sequence.append(pair)
            action_pair_counts[pair] += 1
            expected_subtask = canonical_sub_task(action["left"], action["right"])
            if target.get("sub_task") != expected_subtask:
                issues["noncanonical_vlm_subtask"] += 1
            if segment.get("sub_task") != expected_subtask:
                issues["noncanonical_vla_subtask"] += 1
            keys = ("primary_action_verb", "sub_task", "guide_action", "num_chunks")
            for key in keys:
                if segment.get(key) != target.get(key):
                    issues[f"vlm_vla_{key}_mismatch"] += 1
                if manifest_segment.get(key) != target.get(key):
                    issues[f"vlm_manifest_{key}_mismatch"] += 1
            if (int(segment["start_frame"]), int(segment["end_frame"])) != (
                int(row["segment_start_frame"]),
                int(row["segment_end_frame"]),
            ):
                issues["vlm_vla_boundary_mismatch"] += 1
            if (
                previous_end is not None
                and int(segment["start_frame"]) != previous_end + 1
            ):
                issues["episode_boundary_gap_or_overlap"] += 1
            previous_end = int(segment["end_frame"])
            segment_rows = previous_end - int(segment["start_frame"]) + 1
            length_counts[segment_rows] += 1
            if int(target["num_chunks"]) != math.ceil(segment_rows / 8):
                issues["num_chunks_mismatch"] += 1
            if segment_rows < 8:
                counts["short_segments_lt8"] += 1
                short_action_counts[pair] += 1
                append_example(
                    examples,
                    "short_segments_lt8",
                    {
                        "row_index": row_index,
                        "episode_index": row["episode_index"],
                        "segment_index": segment_index,
                        "task_name": episode["task_name"],
                        "frames": segment_rows,
                        "action": pair,
                    },
                )
            is_terminal = segment_index == len(segments) - 1
            if action == {"left": "still", "right": "still"}:
                counts["both_still_total"] += 1
                counts[
                    "both_still_terminal" if is_terminal else "both_still_nonterminal"
                ] += 1
                if segment_rows < 8:
                    counts["both_still_short"] += 1
            if previous_pair == pair:
                counts["adjacent_same_action_pair"] += 1
                adjacent_same_counts[pair] += 1
                if segment_rows < 8:
                    counts["adjacent_same_with_current_short"] += 1
                append_example(
                    examples,
                    "adjacent_same_action_pair",
                    {
                        "row_index": row_index,
                        "episode_index": row["episode_index"],
                        "segment_index": segment_index,
                        "task_name": episode["task_name"],
                        "action": pair,
                        "current_frames": segment_rows,
                    },
                )
            previous_pair = pair

            if segment_index == 0:
                if row.get("previous_goal") is not None:
                    issues["initial_previous_goal_present"] += 1
            elif row.get("previous_goal") != previous_target:
                issues["history_previous_goal_mismatch"] += 1
            previous_target = target

            match = THINK_RE.fullmatch(str(row["messages"][2]["content"]))
            if match is None:
                issues["assistant_format_mismatch"] += 1
                reasoning = ""
                answer = None
            else:
                reasoning = match.group(1).strip()
                answer = json.loads(match.group(2))
                if answer != target:
                    issues["assistant_target_mismatch"] += 1
            closing = cot_final_action_sentence(
                action["left"], action["right"], int(target["num_chunks"])
            )
            if not reasoning.endswith(closing):
                issues["reasoning_closing_mismatch"] += 1
                body = reasoning
            else:
                body = reasoning[: -len(closing)].strip()
            if body != str(ann.get("reasoning", "")).strip():
                issues["annotation_reasoning_mismatch"] += 1
            body_lower = body.lower()
            digest = hashlib.sha256(body_lower.encode()).hexdigest()
            exact_reasoning_hashes.add(digest)
            words = WORD_RE.findall(body_lower)
            prefix8 = " ".join(words[:8])
            prefix8_counts[prefix8] += 1
            first_sentence = re.split(r"(?<=[.!?])\s+", body, maxsplit=1)[0].strip()
            first_sentence_counts[first_sentence] += 1
            normalized = normalize_opening(first_sentence)
            normalized_first_sentence_counts[normalized] += 1
            sentence_counts[len(re.findall(r"[.!?](?:\s|$)", body))] += 1
            style = str(ann.get("style_name", "unknown"))
            style_counts[style] += 1
            style_openings[style][prefix8] += 1
            phrase_checks = {
                "camera_views_show": r"\bcamera views? (?:show|indicate|provide|reveal)",
                "direct_line_of_sight": r"\bdirect line of sight\b",
                "clear_of_obstruction": r"\bclear of obstruction\b",
                "positioned_to": r"\bpositioned to\b",
                "current_state_shows": r"\bcurrent state (?:shows|indicates)\b",
                "right_arm_remains": r"\bright arm remains\b",
                "left_arm_remains": r"\bleft arm remains\b",
                "task_relevant_target": r"\btask-relevant target\b",
                "cannot_confirm": r"\bcannot confirm\b",
            }
            for name, pattern in phrase_checks.items():
                if re.search(pattern, body_lower):
                    counts[f"cot_phrase_{name}"] += 1
            forbidden_objects = re.findall(
                r"\b(?:bottle|block|can|cup|bowl|shoe|microwave|laptop|switch|"
                r"scanner|box|bin|basket|red|green|blue|yellow|orange|black|white)\b",
                body_lower,
            )
            if forbidden_objects:
                issues["cot_concrete_object_or_color_claim"] += 1
                append_example(
                    examples,
                    "cot_concrete_object_or_color_claim",
                    {
                        "row_index": row_index,
                        "matches": forbidden_objects,
                        "body": body,
                    },
                )

            required = row.get("required_arm")
            if row.get("task_arm_mode") == "single" and required in {"left", "right"}:
                other = "right" if required == "left" else "left"
                if action[other] != "still":
                    issues["single_arm_nonrequired_arm_active"] += 1
                    append_example(
                        examples,
                        "single_arm_nonrequired_arm_active",
                        {
                            "row_index": row_index,
                            "required_arm": required,
                            "action": pair,
                        },
                    )

            for arm, offset in (("left", 0), ("right", 8)):
                metric = endpoint_metrics(row, offset)
                verb = action[arm]
                candidate_names: list[str] = []
                if verb == "still":
                    if metric["translation_m"] >= 0.01:
                        candidate_names.append("still_translation_ge_10mm")
                    if metric["translation_m"] >= 0.02:
                        candidate_names.append("still_translation_ge_20mm")
                    if metric["rotation_rad"] >= 0.05:
                        candidate_names.append("still_rotation_ge_0_05rad")
                    if abs(metric["gripper_delta"]) >= 0.20:
                        candidate_names.append("still_gripper_change_ge_0_20")
                elif verb == "lift" and metric["vertical_m"] <= -0.005:
                    candidate_names.append("lift_endpoint_down_ge_5mm")
                elif verb == "lower" and metric["vertical_m"] >= 0.005:
                    candidate_names.append("lower_endpoint_up_ge_5mm")
                elif verb == "open_gripper" and metric["gripper_delta"] <= 0.05:
                    candidate_names.append("open_gripper_wrong_or_tiny_delta")
                elif verb == "close_gripper" and metric["gripper_delta"] >= -0.05:
                    candidate_names.append("close_gripper_wrong_or_tiny_delta")
                elif verb == "turn" and metric["rotation_rad"] < 0.02:
                    candidate_names.append("turn_endpoint_rotation_lt_0_02rad")
                elif (
                    verb == "move"
                    and metric["translation_m"] < 0.003
                    and metric["rotation_rad"] < 0.02
                ):
                    candidate_names.append("move_endpoint_pose_change_tiny")
                for name in candidate_names:
                    candidates[name] += 1
                    append_example(
                        examples,
                        name,
                        {
                            "row_index": row_index,
                            "episode_index": row["episode_index"],
                            "segment_index": segment_index,
                            "task_name": episode["task_name"],
                            "arm": arm,
                            "action": verb,
                            "frames": segment_rows,
                            "metrics": metric,
                        },
                    )
            raw_segments.append(
                {
                    "row_index": row_index,
                    "segment_index": segment_index,
                    "start": int(segment["start_frame"]),
                    "end": int(segment["end_frame"]),
                    "left": action["left"],
                    "right": action["right"],
                }
            )
            row_index += 1
        task_sequences[str(episode["task_name"])][" | ".join(sequence)] += 1
        raw_jobs.append(
            {
                "episode_index": int(episode["global_episode_index"]),
                "task_name": str(episode["task_name"]),
                "variant": str(episode["source_variant"]),
                "source": str(manifest_episode["source_paths"]["endpose_parquet"]),
                "all_frames": int(episode["all_frames"]),
                "segments": raw_segments,
            }
        )

    for iterator, name in (
        (vlm, "vlm"),
        (manifest, "manifest"),
        (annotation, "annotation"),
    ):
        if next(iterator, None) is not None:
            issues[f"{name}_extra_rows"] += 1

    top = lambda counter, n=20: [
        {"value": key, "count": value} for key, value in counter.most_common(n)
    ]
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "source_modified": False,
        "counts": dict(counts),
        "structural_issues": dict(issues),
        "endpoint_semantic_candidates": dict(candidates),
        "action_pair_counts": dict(action_pair_counts),
        "segment_length": {
            "min": min(length_counts),
            "max": max(length_counts),
            "counts_1_to_16": {str(k): length_counts[k] for k in range(1, 17)},
            "short_lt8": counts["short_segments_lt8"],
            "short_action_top": top(short_action_counts, 30),
        },
        "adjacent_same_action": {
            "count": counts["adjacent_same_action_pair"],
            "with_current_short": counts["adjacent_same_with_current_short"],
            "top": top(adjacent_same_counts, 30),
        },
        "cot_diversity": {
            "styles": dict(style_counts),
            "unique_full_reasoning": len(exact_reasoning_hashes),
            "unique_first_sentences": len(first_sentence_counts),
            "unique_normalized_first_sentences": len(normalized_first_sentence_counts),
            "unique_opening_8_words": len(prefix8_counts),
            "top_first_sentences": top(first_sentence_counts),
            "top_normalized_first_sentences": top(normalized_first_sentence_counts),
            "top_opening_8_words": top(prefix8_counts),
            "sentence_count_distribution": dict(sentence_counts),
            "top_openings_by_style": {
                style: top(counter, 10) for style, counter in style_openings.items()
            },
            "phrase_counts": {
                key.removeprefix("cot_phrase_"): value
                for key, value in counts.items()
                if key.startswith("cot_phrase_")
            },
        },
        "task_sequence_diversity": {
            task: {
                "unique_sequences": len(counter),
                "episodes": sum(counter.values()),
                "top_sequence_count": counter.most_common(1)[0][1],
            }
            for task, counter in sorted(task_sequences.items())
        },
        "examples": dict(examples),
        "seconds": round(time.monotonic() - started, 3),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "structural_semantic_template_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, raw_jobs


def raw_job(job: Mapping[str, Any]) -> dict[str, Any]:
    table = pq.read_table(
        job["source"], columns=["observation.state", "action"], use_threads=False
    )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    action_rows = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    result: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for segment in job["segments"]:
        start, end = segment["start"], segment["end"]
        points = np.concatenate(
            (states[start : start + 1], action_rows[start : end + 1])
        )
        for arm, offset in (("left", 0), ("right", 8)):
            verb = segment[arm]
            xyz = points[:, offset : offset + 3]
            grip = points[:, offset + 7]
            distances = np.linalg.norm(xyz - xyz[0], axis=1)
            rotations = np.asarray(
                [
                    quaternion_angle(points[0, offset + 3 : offset + 7], q)
                    for q in points[:, offset + 3 : offset + 7]
                ]
            )
            dz = float(xyz[-1, 2] - xyz[0, 2])
            metrics = {
                "max_translation_m": float(distances.max()),
                "net_translation_m": float(distances[-1]),
                "path_length_m": float(
                    np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()
                ),
                "max_rotation_rad": float(rotations.max()),
                "net_rotation_rad": float(rotations[-1]),
                "net_vertical_m": dz,
                "min_vertical_m": float((xyz[:, 2] - xyz[0, 2]).min()),
                "max_vertical_m": float((xyz[:, 2] - xyz[0, 2]).max()),
                "gripper_range": float(grip.max() - grip.min()),
                "gripper_delta": float(grip[-1] - grip[0]),
            }
            names: list[str] = []
            if verb == "still":
                if metrics["max_translation_m"] >= 0.01:
                    names.append("still_max_translation_ge_10mm")
                if metrics["max_translation_m"] >= 0.02:
                    names.append("still_max_translation_ge_20mm")
                if metrics["max_rotation_rad"] >= 0.05:
                    names.append("still_max_rotation_ge_0_05rad")
                if metrics["gripper_range"] >= 0.20:
                    names.append("still_gripper_range_ge_0_20")
            elif verb == "turn" and metrics["max_rotation_rad"] < 0.02:
                names.append("turn_max_rotation_lt_0_02rad")
            elif (
                verb == "move"
                and metrics["max_translation_m"] < 0.003
                and metrics["max_rotation_rad"] < 0.02
            ):
                names.append("move_max_pose_change_tiny")
            elif verb == "lift" and dz < -0.001:
                names.append("lift_net_down_gt_1mm")
                if dz < -0.005 and metrics["max_vertical_m"] < 0.001:
                    names.append("lift_only_down_gt_5mm")
            elif verb == "lower" and dz > 0.001:
                names.append("lower_net_up_gt_1mm")
                if dz > 0.005 and metrics["min_vertical_m"] > -0.001:
                    names.append("lower_only_up_gt_5mm")
            elif (
                verb in {"open_gripper", "close_gripper"}
                and metrics["gripper_range"] < 0.01
            ):
                names.append("gripper_action_without_motion")
            for name in names:
                result[name] += 1
                if len(examples) < 30:
                    examples.append(
                        {
                            "candidate": name,
                            "row_index": segment["row_index"],
                            "episode_index": job["episode_index"],
                            "segment_index": segment["segment_index"],
                            "task_name": job["task_name"],
                            "variant": job["variant"],
                            "arm": arm,
                            "action": verb,
                            "metrics": metrics,
                            "source": job["source"],
                        }
                    )
    return {"counts": dict(result), "examples": examples}


def raw_audit(
    jobs: Iterable[dict[str, Any]], out: Path, workers: int
) -> dict[str, Any]:
    started = time.monotonic()
    counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    job_list = list(jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(raw_job, job_list), 1):
            counts.update(result["counts"])
            for example in result["examples"]:
                append_example(examples, example["candidate"], example)
            if index % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "event": "raw_progress",
                            "episodes": index,
                            "total": len(job_list),
                            "candidates": dict(counts),
                        }
                    ),
                    flush=True,
                )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": len(job_list),
        "candidate_counts": dict(counts),
        "examples": dict(examples),
        "seconds": round(time.monotonic() - started, 3),
        "note": "Candidates are numeric screening results, not automatically confirmed semantic errors.",
    }
    (out / "raw_trajectory_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-workers", type=int, default=12)
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()
    structural, jobs = structural_audit(args.root, args.annotations, args.output)
    raw = None if args.skip_raw else raw_audit(jobs, args.output, args.raw_workers)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "structural_semantic_template": structural,
        "raw_trajectory": raw,
    }
    (args.output / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "done": True}, indent=2))


if __name__ == "__main__":
    main()
