"""Production RoboTwin recovery CoT generation in the Embodied format.

This module intentionally keeps recovery data preparation separate from the
successful-demonstration pipeline.  It reads the audited native recovery HDF5
files, derives 16-D end-effector state/action streams, segments the recovery
trajectory, asks a multimodal teacher for causal pre-action reasoning and a
grounded sub-task description, and materializes the exact VLM/VLA training
contracts used by the existing ``robotwin_cot`` release.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import h5py
import numpy as np
from PIL import Image

from atomic_seg.common.artifacts import (
    append_jsonl,
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    bounded_parallel,
    compact_json,
    jsonl_rows,
    repair_jsonl_tail,
    sha256_file,
    utc_now,
)
from atomic_seg.common.model_client import (
    Endpoint,
    non_loopback_endpoints,
    normalize_endpoint,
    request_model,
)
from atomic_seg.common.numeric import round_vector as _round_vector
from atomic_seg.recovery_cot.contracts import (
    ANNOTATION_SCHEMA,
    CAMERAS,
    CHUNK_SIZE,
    CODE_VERSION,
    DEFAULT_REFERENCE_DRIVER,
    FORBIDDEN_VLA_KEYS,
    RANK_SCHEMA,
    STATE_LAYOUT,
    VLA_SCHEMA,
    VLM_FORMAT_VERSION,
)
from atomic_seg.recovery_cot.native import (
    _generic_subtask,
    _primitive_for_arm,
    load_manifest,
    load_native_endpose,
    manifest_paths,
)
from atomic_seg.config import SegmentationConfig
from atomic_seg.recovery_cot.prompts import (
    _image_rows,
    _template_candidate,
    _validate_candidate,
    _validate_critic,
    candidate_prompt,
    critic_prompt,
    image_directory,
)
from atomic_seg.recovery_cot.planner_contract import (
    bind_planner_contract,
    validate_planner_row,
)
from atomic_seg.recovery_cot.policy import QUALITY_POLICY, validate_segments
from atomic_seg.recovery_cot.quality import (
    DiversityAudit,
    annotation_quality,
    complete_reasoning,
)


def _walk_keys(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def load_reference_driver(path: Path = DEFAULT_REFERENCE_DRIVER) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"reference driver not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "robotwin_reference_adapter",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load reference driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return bind_planner_contract(module)


def runtime_fingerprint() -> str:
    """Manual rank launches get the same code-change protection as the runner."""
    digest = hashlib.sha256()
    root = Path(__file__).resolve().parents[1]
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def shard_paths(output_root: Path, rank: int) -> dict[str, Path]:
    suffix = f"rank_{rank:02d}"
    return {
        "source": output_root / "shards/source" / f"{suffix}.jsonl",
        "annotations": output_root / "shards/annotations" / f"{suffix}.jsonl",
        "vlm": output_root / "shards/vlm" / f"{suffix}.jsonl",
        "vla": output_root / "shards/vla" / f"{suffix}.jsonl",
        "prepare_report": output_root / "shards/reports" / f"{suffix}.prepare.json",
        "generation_report": output_root
        / "shards/reports"
        / f"{suffix}.generation.json",
        "done": output_root / "control" / f"{suffix}.done.json",
        "failures": output_root / "control" / f"{suffix}.failures.json",
        "raw": output_root / "logs/raw" / f"{suffix}.jsonl",
    }


def materialize_episode_images(
    *,
    output_root: Path,
    record: Mapping[str, Any],
    hdf5_path: Path,
    frames: Iterable[int],
    image_width: int,
    jpeg_quality: int,
) -> None:
    directory = image_directory(output_root, record)
    directory.mkdir(parents=True, exist_ok=True)
    wanted = sorted(set(int(value) for value in frames))
    with h5py.File(hdf5_path, "r") as source:
        for camera, key in CAMERAS:
            dataset = source[key]
            for frame in wanted:
                destination = directory / f"{camera}_f{frame:06d}.jpg"
                if destination.is_file() and destination.stat().st_size > 0:
                    continue
                encoded = bytes(np.asarray(dataset[frame], dtype=np.uint8))
                image = Image.open(io.BytesIO(encoded)).convert("RGB")
                if image_width > 0 and image.width != image_width:
                    height = max(1, round(image.height * image_width / image.width))
                    image = image.resize(
                        (image_width, height),
                        Image.Resampling.LANCZOS,
                    )
                temporary = destination.with_name(
                    f".{destination.name}.tmp-{os.getpid()}-{threading.get_ident()}"
                )
                image.save(
                    temporary,
                    format="JPEG",
                    quality=jpeg_quality,
                    optimize=True,
                )
                os.replace(temporary, destination)


def build_source_record(
    manifest: Mapping[str, Any],
    *,
    output_root: Path,
    image_width: int,
    jpeg_quality: int,
) -> dict[str, Any]:
    hdf5_path = Path(str(manifest["source_paths"]["recovery_hdf5"]))
    states, actions, _timestamps, _extras = load_native_endpose(hdf5_path)
    if len(states) != int(manifest["all_frames"]):
        raise ValueError(f"{manifest['episode_id']}: frame count changed")
    steps: list[dict[str, Any]] = []
    validate_segments(manifest["segments"])
    for source in manifest["segments"]:
        start = int(source["start_frame"])
        end = int(source["end_frame"])
        left = str(source["left_action"])
        right = str(source["right_action"])
        step = {
            "step_index": len(steps) + 1,
            "source_segment_id": len(steps),
            "global_block_index": int(source["global_block_index"]),
            "start_frame": start,
            "end_frame": end,
            "left_action": left,
            "right_action": right,
            "primary_action_verb": f"left:{left}, right:{right}",
            "source_sub_task": _generic_subtask(left, right),
            "action_description": _generic_subtask(left, right),
            "num_chunks": int(math.ceil((end - start + 1) / CHUNK_SIZE)),
            "guide_action": _round_vector(actions[end] - states[start]),
            "guide_action_semantics": (
                "action[segment_end_frame_inclusive]-"
                "observation.state[segment_start_frame]"
            ),
            "native_group_index": int(source["native_group_index"]),
            "native_semantic_action": str(source["native_semantic_action"]),
            "native_event_stage": str(source["native_event_stage"]),
            "training_eligible": True,
            "quarantine_reasons": [],
            "source_atomic_spans": copy.deepcopy(source.get("source_atomic_spans", [])),
            "short_segment_reason": source.get("short_segment_reason"),
            "quality_policy": QUALITY_POLICY,
        }
        steps.append(step)
    materialize_episode_images(
        output_root=output_root,
        record=manifest,
        hdf5_path=hdf5_path,
        frames=[int(step["start_frame"]) for step in steps],
        image_width=image_width,
        jpeg_quality=jpeg_quality,
    )
    instruction = str(manifest["task_instruction"])
    return {
        "schema_version": "robotwin_recovery_v28_source_episode_v1",
        "code_version": CODE_VERSION,
        "index": int(manifest["global_episode_index"]),
        "episode_id": str(manifest["episode_id"]),
        "episode_class": "recovery_success",
        "failure_type": str(manifest["failure_type"]),
        "source_dataset": "RoboTwin-Recovery",
        "source_case_type": "audited_failure_recovery",
        "source_variant": str(manifest["source_variant"]),
        "task_index": int(manifest["task_index"]),
        "task_name": str(manifest["task_name"]),
        "official_episode_index": int(manifest["official_episode_index"]),
        "global_episode_index": int(manifest["global_episode_index"]),
        "training_episode_index": int(manifest["training_episode_index"]),
        "task_instruction": instruction,
        "instruction_contract": {
            "selected": instruction,
            "source": "recovery_rollout_metadata",
            "selection_confirmed": True,
            "resolution_type": "exact_recorded_instruction",
            "exact_rollout_prompt": instruction,
            "exact_rollout_prompt_confirmed": True,
            "semantic_task_confirmed": True,
            "training_instruction_eligible": True,
        },
        "outcome_label": "verified_recovery_success",
        "timeline_contract": "recovery_only_from_verified_failure_state",
        "recovery_level": int(manifest["recovery_level"]),
        "recovery_level_name": str(manifest["recovery_level_name"]),
        "all_frames": int(manifest["all_frames"]),
        "total_steps": len(steps),
        "source_steps": steps,
        "source_paths": copy.deepcopy(manifest["source_paths"]),
        "training_eligible_blocks": len(steps),
        "quarantine_blocks": 0,
        "format_contract": {
            "episode_role": "audit_and_grouping_container",
            "policy_learning_unit": "one_recovery_atomic_block",
            "training_input_visibility": "current_boundary_and_completed_history_only",
            "post_action_fields": "audit_only",
            "cot_embedded_in_training_assistant": True,
        },
        "source_identity_validation": {
            "pair_audit_passed": True,
            "source_failure_verified": True,
            "recovery_success": True,
            "native_joint_state_dim": 14,
            "derived_endpose_state_dim": 16,
            "endpose_action_alignment": "next_observation_state",
        },
        "generation": {
            "method": "pending_qwen38_recovery_causal_pre_action_generation",
            "fresh_model_call": False,
            "raw_hidden_reasoning_included": False,
        },
    }


def prepare_rank(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    workers: int,
    image_width: int,
    jpeg_quality: int,
    resume: bool,
    limit_episodes: int | None = None,
) -> list[dict[str, Any]]:
    manifest_rows, _summary = load_manifest(output_root)
    selected = [
        row
        for row in manifest_rows
        if int(row["global_episode_index"]) % world_size == rank
    ]
    if limit_episodes is not None:
        selected = selected[: int(limit_episodes)]
    paths = shard_paths(output_root, rank)
    paths["annotations"].parent.mkdir(parents=True, exist_ok=True)
    paths["raw"].parent.mkdir(parents=True, exist_ok=True)
    if resume:
        repair_jsonl_tail(paths["source"])
        existing_rows = jsonl_rows(paths["source"])
    else:
        existing_rows = []
    for row in existing_rows:
        if row.get("code_version") != CODE_VERSION:
            raise ValueError(f"{paths['source']}: code version mismatch")
    existing = {int(row["global_episode_index"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise ValueError("duplicate cached recovery source episodes")
    selected_by_index = {int(row["global_episode_index"]): row for row in selected}
    if not set(existing) <= set(selected_by_index):
        raise ValueError("cached source episodes do not belong to this recovery rank")
    for index, record in existing.items():
        manifest = selected_by_index[index]
        if len(record["source_steps"]) != len(manifest["segments"]):
            raise ValueError("cached source/manifest segment mismatch")
        for step, segment in zip(record["source_steps"], manifest["segments"]):
            for field in (
                "start_frame",
                "end_frame",
                "left_action",
                "right_action",
                "global_block_index",
            ):
                if step[field] != segment[field]:
                    raise ValueError(f"cached source/manifest mismatch: {field}")
    pending = [
        row for row in selected if int(row["global_episode_index"]) not in existing
    ]
    paths["source"].parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and paths["source"].exists() else "w"
    failures: list[dict[str, Any]] = []
    started = time.monotonic()

    def work(row: Mapping[str, Any]) -> dict[str, Any]:
        return build_source_record(
            row,
            output_root=output_root,
            image_width=image_width,
            jpeg_quality=jpeg_quality,
        )

    with paths["source"].open(mode, encoding="utf-8", buffering=1) as stream:
        completed = 0
        for source, record, error in bounded_parallel(
            pending,
            workers=workers,
            fn=work,
        ):
            index = int(source["global_episode_index"])
            if error is not None:
                failures.append(
                    {
                        "global_episode_index": index,
                        "episode_id": source["episode_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(
                    f"[prepare rank={rank}] FAILED episode={index}: {error}",
                    flush=True,
                )
                continue
            assert record is not None
            append_jsonl(stream, record)
            existing[index] = record
            completed += 1
            if completed == 1 or completed % 25 == 0:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"[prepare rank={rank}] {completed}/{len(pending)} new, "
                    f"{len(existing)}/{len(selected)} total, "
                    f"{completed / elapsed:.2f} episodes/s",
                    flush=True,
                )
    report = {
        "schema_version": "robotwin_recovery_v28_rank_prepare_v1",
        "code_version": CODE_VERSION,
        "created_at_utc": utc_now(),
        "rank": rank,
        "world_size": world_size,
        "episodes": len(existing),
        "blocks": sum(int(row["total_steps"]) for row in existing.values()),
        "failures": failures,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    atomic_write_json(paths["prepare_report"], report)
    if failures:
        atomic_write_json(paths["failures"], {"stage": "prepare", **report})
        raise RuntimeError(f"rank {rank}: {len(failures)} preparation failures")
    ordered = [existing[int(row["global_episode_index"])] for row in selected]
    atomic_write_jsonl(paths["source"], ordered)
    return ordered


def annotate_one(
    *,
    output_root: Path,
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    previous_step: Mapping[str, Any] | None,
    states: np.ndarray,
    endpoints: Sequence[Endpoint],
    model: str,
    timeout: float,
    max_tokens: int,
    critic_max_tokens: int,
    max_attempts: int,
    revision_rounds: int,
    seed_salt: str,
    raw_log: Any,
    template: bool,
) -> dict[str, Any]:
    images = _image_rows(output_root, record, step, previous_step)
    paths = [output_root / str(row["path"]) for row in images]
    roles = [str(row["role"]) for row in images]
    if template:
        candidate = _template_candidate(record, step)
        candidate["reasoning"] = complete_reasoning(candidate["reasoning"], step)
        critic = {
            "verdict": "PASS",
            "checks": {
                "concrete_sub_task": True,
                "current_observation_grounded": True,
                "history_grounded": True,
                "no_future_leakage": True,
                "no_unsupported_effect": True,
                "primitive_match": True,
                "reasoning_supports_action": True,
            },
            "issues": [],
            "revision_instructions": [],
        }
        return {
            "schema_version": ANNOTATION_SCHEMA,
            "episode_id": record["episode_id"],
            "segment_id": int(step["source_segment_id"]),
            "step_index": int(step["step_index"]),
            "current_frame": int(step["start_frame"]),
            "visible_images": images,
            "frozen_action": {
                key: step[key]
                for key in (
                    "left_action",
                    "right_action",
                    "primary_action_verb",
                    "guide_action",
                    "num_chunks",
                )
            },
            "training_eligible": True,
            "quarantine_reasons": [],
            "candidate": candidate,
            "critic_1": critic,
            "critic_2": copy.deepcopy(critic),
            "revision_history": [],
            "visibility_contract": {
                "all_image_frames_lte_current": True,
                "end_image_visible": False,
                "end_state_visible": False,
                "future_action_visible": False,
                "episode_outcome_visible": False,
                "action_primitive_visible_to_annotator": True,
                "numeric_action_target_visible_to_annotator": False,
                "action_target_visible_to_training_human": False,
            },
            "generation": {"model": "template", "method": "offline_template"},
            "critic_generation": {},
        }

    base_seed = int.from_bytes(
        hashlib.sha256(
            f"{record['episode_id']}\0{step['source_segment_id']}\0{seed_salt}".encode()
        ).digest()[:4],
        "big",
    )
    endpoint_start = base_seed % len(endpoints)
    errors: list[str] = []
    revision_history: list[dict[str, Any]] = []
    feedback: list[str] = []
    request_number = 0

    def call(
        *,
        system: str,
        prompt: str,
        token_limit: int,
        temperature: float,
        enable_thinking: bool,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal request_number
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            endpoint = endpoints[
                (endpoint_start + request_number + attempt) % len(endpoints)
            ]
            try:
                output, metadata = request_model(
                    endpoint,
                    model=model,
                    system=system,
                    prompt=prompt,
                    image_paths=paths,
                    image_roles=roles,
                    timeout=timeout,
                    max_tokens=token_limit,
                    temperature=temperature,
                    seed=(base_seed + request_number * 1009 + attempt) % (2**31),
                    enable_thinking=enable_thinking,
                )
                if enable_thinking and not (
                    metadata.get("hidden_reasoning_present") is True
                    and int(metadata.get("hidden_reasoning_words", 0) or 0) > 0
                    and int(metadata.get("usage", {}).get("reasoning_tokens", 0) or 0)
                    > 1
                    and bool(metadata.get("hidden_reasoning_sha256"))
                ):
                    raise RuntimeError(
                        "candidate response lacks substantive hidden reasoning"
                    )
                raw_log.write(
                    {
                        "created_at_utc": utc_now(),
                        "episode_id": record["episode_id"],
                        "segment_id": step["source_segment_id"],
                        "request_number": request_number,
                        "attempt": attempt + 1,
                        "status": "ok",
                        **metadata,
                    }
                )
                request_number += 1
                return output, metadata
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                errors.append(f"{endpoint.base_url}: {type(exc).__name__}: {exc}")
                raw_log.write(
                    {
                        "created_at_utc": utc_now(),
                        "episode_id": record["episode_id"],
                        "segment_id": step["source_segment_id"],
                        "request_number": request_number,
                        "attempt": attempt + 1,
                        "endpoint": endpoint.base_url,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                time.sleep(min(10.0, 1.5 * (attempt + 1)))
        raise RuntimeError("all model attempts failed") from last_error

    for revision in range(revision_rounds + 1):
        raw_candidate, candidate_meta = call(
            system=(
                "You produce causal pre-action annotations for a dual-arm robot. "
                "Use only current and completed-history observations. Return "
                "strict JSON and never reveal future trajectory information. "
                "Before returning JSON, perform substantive private internal "
                "reasoning even when the requested primitive appears simple."
            ),
            prompt=candidate_prompt(
                record=record,
                step=step,
                previous_step=previous_step,
                states=states,
                revision_feedback=feedback,
            ),
            token_limit=max_tokens,
            temperature=0.2,
            # Hidden reasoning is not stored in training data, but it improves
            # semantic grounding on difficult recovery states.  The explicit
            # validated ``reasoning`` field remains the only saved CoT.
            enable_thinking=True,
        )
        try:
            candidate = _validate_candidate(raw_candidate, step=step)
        except Exception as exc:  # noqa: BLE001
            feedback = [str(exc)]
            revision_history.append(
                {"revision": revision, "candidate": raw_candidate, "issues": feedback}
            )
            continue
        critics: list[dict[str, Any]] = []
        critic_meta: list[dict[str, Any]] = []
        for critic_index in (1, 2):
            raw_critic, metadata = call(
                system=(
                    "You are an independent strict critic for causal robot "
                    "pre-action annotations. Return JSON only."
                ),
                prompt=critic_prompt(
                    record=record,
                    step=step,
                    previous_step=previous_step,
                    states=states,
                    candidate=candidate,
                    critic_index=critic_index,
                ),
                token_limit=critic_max_tokens,
                temperature=0.0,
                enable_thinking=False,
            )
            critics.append(_validate_critic(raw_critic))
            critic_meta.append(metadata)
        if all(value["verdict"] == "PASS" for value in critics):
            return {
                **annotation_quality(candidate, step),
                "schema_version": ANNOTATION_SCHEMA,
                "episode_id": record["episode_id"],
                "segment_id": int(step["source_segment_id"]),
                "step_index": int(step["step_index"]),
                "current_frame": int(step["start_frame"]),
                "visible_images": images,
                "frozen_action": {
                    key: step[key]
                    for key in (
                        "left_action",
                        "right_action",
                        "primary_action_verb",
                        "guide_action",
                        "num_chunks",
                    )
                },
                "training_eligible": True,
                "quarantine_reasons": [],
                "candidate": candidate,
                "critic_1": critics[0],
                "critic_2": critics[1],
                "revision_history": revision_history,
                "visibility_contract": {
                    "all_image_frames_lte_current": True,
                    "end_image_visible": False,
                    "end_state_visible": False,
                    "future_action_visible": False,
                    "episode_outcome_visible": False,
                    "action_primitive_visible_to_annotator": True,
                    "numeric_action_target_visible_to_annotator": False,
                    "action_target_visible_to_training_human": False,
                },
                "generation": {
                    "model": model,
                    "method": "qwen38_recovery_causal_pre_action_generation",
                    "revision_count": revision,
                    "request": candidate_meta,
                    "hidden_reasoning_used_for_generation": True,
                    "errors": errors,
                    "raw_hidden_reasoning_included": False,
                },
                "critic_generation": {
                    "critic_1": critic_meta[0],
                    "critic_2": critic_meta[1],
                },
            }
        feedback = [
            instruction
            for critic in critics
            for instruction in critic["revision_instructions"]
        ]
        revision_history.append(
            {
                "revision": revision,
                "candidate": candidate,
                "critic_1": critics[0],
                "critic_2": critics[1],
            }
        )
    raise RuntimeError(
        "candidate did not pass both critics after revisions: " + compact_json(feedback)
    )


class JsonlLog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("a", encoding="utf-8", buffering=1)
        self.lock = threading.Lock()

    def write(self, value: Mapping[str, Any]) -> None:
        with self.lock:
            self.stream.write(compact_json(value) + "\n")
            self.stream.flush()

    def close(self) -> None:
        with self.lock:
            self.stream.close()


def annotation_key(row: Mapping[str, Any]) -> int:
    """Return the manifest-global identity for one atomic annotation.

    Recovery rollout names are not guaranteed to be unique: the source audit
    can contain multiple official episodes whose recovery directory has the
    same basename. ``global_block_index`` is assigned once by the manifest and
    is therefore the only collision-free annotation key across ranks.
    """

    return int(row["global_block_index"])


def step_annotation_key(step: Mapping[str, Any]) -> int:
    return int(step["global_block_index"])


def candidate_used_hidden_reasoning(annotation: Mapping[str, Any]) -> bool:
    """Require substantive hidden model reasoning, not a compatibility flag."""

    request = annotation.get("generation", {}).get("request", {})
    return (
        request.get("hidden_reasoning_present") is True
        and int(request.get("hidden_reasoning_words", 0) or 0) > 0
        and int(request.get("usage", {}).get("reasoning_tokens", 0) or 0) > 1
        and bool(request.get("hidden_reasoning_sha256"))
    )


def _generation_jobs(
    records: Sequence[Mapping[str, Any]],
    existing: Mapping[int, Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    jobs: list[Mapping[str, Any]] = []
    for record in records:
        if any(
            step_annotation_key(step) not in existing for step in record["source_steps"]
        ):
            jobs.append(record)
    return jobs


def generate_rank_annotations(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    records: Sequence[Mapping[str, Any]],
    endpoint_values: Sequence[str],
    model: str,
    workers: int,
    timeout: float,
    max_tokens: int,
    critic_max_tokens: int,
    max_attempts: int,
    revision_rounds: int,
    generation_rounds: int,
    seed_salt: str,
    resume: bool,
    template: bool,
) -> dict[int, dict[str, Any]]:
    paths = shard_paths(output_root, rank)
    # A failure report describes the previous exhausted launch attempt, not the
    # health of the newly resumed worker. Clear it as soon as the next worker
    # starts; a fresh report is written atomically if this attempt also ends with
    # pending blocks.
    paths["failures"].unlink(missing_ok=True)
    if resume:
        repair_jsonl_tail(paths["annotations"])
        existing_rows = jsonl_rows(paths["annotations"])
    else:
        existing_rows = []
    existing = {
        annotation_key(row): row
        for row in existing_rows
        if candidate_used_hidden_reasoning(row)
        and row.get("quality_policy") == QUALITY_POLICY
    }
    if len({annotation_key(row) for row in existing_rows}) != len(existing_rows):
        raise ValueError("duplicate cached recovery annotations")
    expected_keys = {
        step_annotation_key(step)
        for record in records
        for step in record["source_steps"]
    }
    if not set(existing) <= expected_keys:
        raise ValueError("cached annotations do not belong to this recovery rank")
    for record in records:
        for step in record["source_steps"]:
            key = step_annotation_key(step)
            if key in existing:
                value = existing[key]
                checked = annotation_quality(value["candidate"], step)
                if any(value.get(k) != v for k, v in checked.items()):
                    raise ValueError(f"stale or invalid recovery annotation {key}")
                if value["candidate"]["sub_task"] != step["source_sub_task"]:
                    raise ValueError(f"cached annotation {key}: sub_task mismatch")
                if any(
                    _validate_critic(value[name])["verdict"] != "PASS"
                    for name in ("critic_1", "critic_2")
                ):
                    raise ValueError(f"cached annotation {key}: critic did not pass")
    endpoints = [normalize_endpoint(value) for value in endpoint_values]
    if not endpoints and not template:
        raise ValueError("at least one endpoint is required")
    raw_log = JsonlLog(paths["raw"])
    failures: list[dict[str, Any]] = []
    started = time.monotonic()
    persist_lock = threading.Lock()
    try:
        for generation_round in range(generation_rounds):
            jobs = _generation_jobs(records, existing)
            if not jobs:
                break

            def work(record: Mapping[str, Any]) -> dict[str, Any]:
                states, _actions, _timestamps, _extras = load_native_endpose(
                    Path(str(record["source_paths"]["recovery_hdf5"]))
                )
                with persist_lock:
                    local = {
                        key: value
                        for key, value in existing.items()
                        if int(value["global_episode_index"])
                        == int(record["global_episode_index"])
                    }
                generated: list[dict[str, Any]] = []
                for index, step in enumerate(record["source_steps"]):
                    key = step_annotation_key(step)
                    if key in local:
                        continue
                    previous = None
                    if index:
                        previous_source = record["source_steps"][index - 1]
                        previous_key = step_annotation_key(previous_source)
                        if previous_key not in local:
                            raise RuntimeError(
                                f"missing preceding annotation {previous_key}"
                            )
                        previous = {
                            **previous_source,
                            "source_sub_task": str(
                                local[previous_key]["candidate"]["sub_task"]
                            ),
                        }
                    annotation = annotate_one(
                        output_root=output_root,
                        record=record,
                        step=step,
                        previous_step=previous,
                        states=states,
                        endpoints=endpoints,
                        model=model,
                        timeout=timeout,
                        max_tokens=max_tokens,
                        critic_max_tokens=critic_max_tokens,
                        max_attempts=max_attempts,
                        revision_rounds=revision_rounds,
                        seed_salt=f"{seed_salt}-round-{generation_round}",
                        raw_log=raw_log,
                        template=template,
                    )
                    annotation.update(
                        {
                            "code_version": CODE_VERSION,
                            "global_episode_index": record["global_episode_index"],
                            "global_block_index": step["global_block_index"],
                            "source_variant": record["source_variant"],
                            "task_name": record["task_name"],
                            "official_episode_index": record["official_episode_index"],
                        }
                    )
                    local[key] = annotation
                    generated.append(annotation)
                    with persist_lock:
                        append_jsonl(stream, annotation)
                        existing[key] = annotation
                return {"annotations": generated}

            completed = 0
            round_failures: list[dict[str, Any]] = []
            with paths["annotations"].open(
                "a" if resume or paths["annotations"].exists() else "w",
                encoding="utf-8",
                buffering=1,
            ) as stream:
                for record, result, error in bounded_parallel(
                    jobs, workers=workers, fn=work
                ):
                    if error is not None:
                        detail = {
                            "episode_id": record["episode_id"],
                            "round": generation_round,
                            "error": f"{type(error).__name__}: {error}",
                        }
                        round_failures.append(detail)
                        print(
                            f"[generate rank={rank}] FAILED episode="
                            f"{record['episode_id']}: {detail['error']}",
                            flush=True,
                        )
                        continue
                    assert result is not None
                    for annotation in result["annotations"]:
                        completed += 1
                        if completed == 1 or completed % 10 == 0:
                            elapsed = max(time.monotonic() - started, 1e-6)
                            print(
                                f"[generate rank={rank}] "
                                f"round={generation_round} {completed} new "
                                f"blocks, {len(existing)} total, "
                                f"{completed / elapsed:.3f} blocks/s",
                                flush=True,
                            )
            failures.extend(round_failures)
            if not round_failures:
                break
    finally:
        raw_log.close()
    pending = _generation_jobs(records, existing)
    ordered = sorted(existing.values(), key=lambda row: int(row["global_block_index"]))
    atomic_write_jsonl(paths["annotations"], ordered)
    if pending:
        atomic_write_json(
            paths["failures"],
            {
                "stage": "generation",
                "rank": rank,
                "world_size": world_size,
                "pending_blocks": len(pending),
                "recent_failures": failures[-200:],
            },
        )
        raise RuntimeError(f"rank {rank}: {len(pending)} blocks remain")
    return existing


def _vla_segment(
    record: Mapping[str, Any],
    step: Mapping[str, Any],
) -> dict[str, Any]:
    start = int(step["start_frame"])
    end = int(step["end_frame"])
    sample_id = vla_sample_id(record, step)
    instruction = str(record["task_instruction"])
    return {
        "sample_id": sample_id,
        "source_segment_id": int(step["source_segment_id"]),
        "source_step_index": int(step["step_index"]),
        "source_total_steps": int(record["total_steps"]),
        "action_name": instruction,
        "task_instruction": instruction,
        "instruction": instruction,
        "instruction_contract": copy.deepcopy(record["instruction_contract"]),
        "sub_task": str(step["source_sub_task"]),
        "primary_action_verb": str(step["primary_action_verb"]),
        "left_action": str(step["left_action"]),
        "right_action": str(step["right_action"]),
        "start_frame": start,
        "end_frame": end,
        "timestamp_formatted": f"{start} - {end}",
        "guide_action": list(step["guide_action"]),
        "guide_action_semantics": str(step["guide_action_semantics"]),
        "num_chunks": int(step["num_chunks"]),
        "training_eligible": True,
        "quarantine_reasons": [],
        "format_contract": {
            "modality": "vla",
            "cot_present": False,
            "action_supervision_only": True,
        },
        "global_block_index": int(step["global_block_index"]),
        "source_variant": str(record["source_variant"]),
        "task_name": str(record["task_name"]),
        "official_episode_index": int(record["official_episode_index"]),
        "global_episode_index": int(record["global_episode_index"]),
        "training_episode_index": int(record["training_episode_index"]),
        "step_index": int(step["step_index"]),
        "total_steps": int(record["total_steps"]),
        "source_global_block_index": int(step["global_block_index"]),
    }


def vla_sample_id(
    record: Mapping[str, Any],
    step: Mapping[str, Any],
) -> str:
    """Build a stable unique sample ID without changing the existing field shape."""

    episode_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(record["episode_id"]))
    return (
        f"{episode_slug}"
        f"__episode_{int(record['global_episode_index']):06d}"
        f"__segment_{int(step['source_segment_id']):03d}"
    )


def materialize_rank(
    *,
    output_root: Path,
    rank: int,
    records: Sequence[Mapping[str, Any]],
    annotations: Mapping[int, Mapping[str, Any]],
    reference_driver: Path,
) -> tuple[int, int]:
    reference = load_reference_driver(reference_driver)
    paths = shard_paths(output_root, rank)
    vlm_rows: list[dict[str, Any]] = []
    vla_rows: list[dict[str, Any]] = []
    for source_record in sorted(
        records, key=lambda row: int(row["global_episode_index"])
    ):
        record = copy.deepcopy(source_record)
        row_record = {
            **record,
            "source_steps": record["source_steps"],
        }
        states, actions, _timestamps, _extras = load_native_endpose(
            Path(str(record["source_paths"]["recovery_hdf5"]))
        )
        segments: list[dict[str, Any]] = []
        for index, original_step in enumerate(record["source_steps"]):
            step = copy.deepcopy(original_step)
            key = step_annotation_key(step)
            annotation = annotations[key]
            candidate = annotation["candidate"]
            row = reference.build_embodied_cot_row(
                output_root=output_root,
                record=row_record,
                step=step,
                previous_step=(record["source_steps"][index - 1] if index else None),
                annotation={"candidate": {"reasoning": candidate["reasoning"]}},
                states=states,
                actions=actions,
            )
            reference.validate_embodied_cot_row(row)
            validate_planner_row(row)
            vlm_rows.append(row)
            segments.append(_vla_segment(record, step))
        for position, segment in enumerate(segments, 1):
            segment["step_index"] = position
            segment["total_steps"] = len(segments)
        instruction = str(record["task_instruction"])
        episode = {
            "schema_version": VLA_SCHEMA,
            "source_episode_id": str(record["episode_id"]),
            "source_dataset": "RoboTwin-Recovery",
            "source_variant": str(record["source_variant"]),
            "task_index": int(record["task_index"]),
            "task_name": str(record["task_name"]),
            "official_episode_index": int(record["official_episode_index"]),
            "global_episode_index": int(record["global_episode_index"]),
            "training_episode_index": int(record["training_episode_index"]),
            "episode_class": "recovery_success",
            "failure_type": str(record["failure_type"]),
            "task": instruction,
            "task_instruction": instruction,
            "instruction": instruction,
            "instruction_contract": copy.deepcopy(record["instruction_contract"]),
            "all_frames": int(record["all_frames"]),
            "source_total_steps": int(record["total_steps"]),
            "total_steps": len(segments),
            "state_layout": list(STATE_LAYOUT),
            "format_contract": {
                "modality": "vla",
                "cot_present": False,
                "task_chain_present": False,
                "action_supervision_only": True,
                "training_only": True,
            },
            "segments": segments,
        }
        forbidden = FORBIDDEN_VLA_KEYS & set(_walk_keys(episode))
        if forbidden:
            raise ValueError(f"VLA contains CoT fields: {sorted(forbidden)}")
        vla_rows.append(episode)
    atomic_write_jsonl(
        paths["vlm"],
        sorted(
            vlm_rows,
            key=lambda row: (int(row["episode_index"]), int(row["segment_index"])),
        ),
    )
    atomic_write_jsonl(
        paths["vla"],
        sorted(vla_rows, key=lambda row: int(row["global_episode_index"])),
    )
    return len(vlm_rows), len(vla_rows)


def run_rank(
    *,
    output_root: Path,
    rank: int,
    world_size: int,
    endpoint_values: Sequence[str],
    model: str,
    prepare_workers: int,
    generation_workers: int,
    image_width: int,
    jpeg_quality: int,
    timeout: float,
    max_tokens: int,
    critic_max_tokens: int,
    max_attempts: int,
    revision_rounds: int,
    generation_rounds: int,
    launch_attempt: int,
    resume: bool,
    reference_driver: Path,
    limit_episodes: int | None = None,
    template: bool = False,
) -> dict[str, Any]:
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world size")
    paths = shard_paths(output_root, rank)
    context_path = output_root / "control" / f"rank_{rank:02d}.context.json"
    context = {
        "code_version": CODE_VERSION,
        "runtime_fingerprint": runtime_fingerprint(),
        "quality_policy": QUALITY_POLICY,
        "rank": rank,
        "world_size": world_size,
        "manifest_sha256": sha256_file(manifest_paths(output_root)[0]),
        "reference_driver_sha256": sha256_file(reference_driver),
        "template": template,
        "model": model,
        "image_width": image_width,
        "jpeg_quality": jpeg_quality,
        "limit_episodes": limit_episodes,
    }
    if context_path.exists():
        if not resume or json.loads(context_path.read_text()) != context:
            raise ValueError(
                "recovery rank inputs/config changed; use a fresh output root"
            )
    elif any(paths[key].exists() for key in ("source", "annotations", "done")):
        raise ValueError(
            "unfingerprinted recovery checkpoints; use a fresh output root"
        )
    atomic_write_json(context_path, context)
    if resume and paths["done"].is_file():
        report = json.loads(paths["done"].read_text(encoding="utf-8"))
        if (
            report.get("code_version") == CODE_VERSION
            and int(report.get("world_size", -1)) == world_size
        ):
            if report.get("manifest_sha256") != sha256_file(
                manifest_paths(output_root)[0]
            ):
                raise ValueError(
                    "completed rank manifest changed; use a fresh output root"
                )
            if report.get("reference_driver_sha256") != sha256_file(reference_driver):
                raise ValueError(
                    "completed rank reference driver changed; use a fresh output root"
                )
            if all(
                paths[key].is_file()
                and sha256_file(paths[key]) == report["hashes"][key]
                for key in ("source", "annotations", "vlm", "vla")
            ):
                return report
            raise ValueError(
                "completed recovery rank files changed; use a fresh output root"
            )
    records = prepare_rank(
        output_root=output_root,
        rank=rank,
        world_size=world_size,
        workers=prepare_workers,
        image_width=image_width,
        jpeg_quality=jpeg_quality,
        resume=resume,
        limit_episodes=limit_episodes,
    )
    annotations = generate_rank_annotations(
        output_root=output_root,
        rank=rank,
        world_size=world_size,
        records=records,
        endpoint_values=endpoint_values,
        model=model,
        workers=generation_workers,
        timeout=timeout,
        max_tokens=max_tokens,
        critic_max_tokens=critic_max_tokens,
        max_attempts=max_attempts,
        revision_rounds=revision_rounds,
        generation_rounds=generation_rounds,
        seed_salt=f"rank-{rank}-launch-{launch_attempt}",
        resume=resume,
        template=template,
    )
    vlm_count, vla_count = materialize_rank(
        output_root=output_root,
        rank=rank,
        records=records,
        annotations=annotations,
        reference_driver=reference_driver,
    )
    report = {
        "schema_version": RANK_SCHEMA,
        "code_version": CODE_VERSION,
        "vlm_format_version": VLM_FORMAT_VERSION,
        "created_at_utc": utc_now(),
        "rank": rank,
        "world_size": world_size,
        "episodes": len(records),
        "blocks": sum(int(row["total_steps"]) for row in records),
        "annotations": len(annotations),
        "vlm_rows": vlm_count,
        "vla_episodes": vla_count,
        "model": "template" if template else model,
        "runtime_fingerprint": runtime_fingerprint(),
        "release_eligible": not template,
        "manifest_sha256": sha256_file(manifest_paths(output_root)[0]),
        "reference_driver_sha256": sha256_file(reference_driver),
        "endpoints": list(endpoint_values),
        "shards": {
            key: str(paths[key]) for key in ("source", "annotations", "vlm", "vla")
        },
        "hashes": {
            key: sha256_file(paths[key])
            for key in ("source", "annotations", "vlm", "vla")
        },
    }
    atomic_write_json(paths["generation_report"], report)
    paths["failures"].unlink(missing_ok=True)
    atomic_write_json(paths["done"], report)
    return report


def _load_rank_reports(output_root: Path, world_size: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for rank in range(world_size):
        path = shard_paths(output_root, rank)["done"]
        if not path.is_file():
            raise FileNotFoundError(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("code_version") != CODE_VERSION:
            raise ValueError(f"rank {rank}: code version mismatch")
        if (
            int(report.get("rank", -1)) != rank
            or int(report.get("world_size", -1)) != world_size
        ):
            raise ValueError(f"rank {rank}: identity mismatch")
        for key in ("source", "annotations", "vlm", "vla"):
            path_value = Path(str(report["shards"][key]))
            if report["hashes"][key] != sha256_file(path_value):
                raise ValueError(f"rank {rank}: {key} hash mismatch")
        reports.append(report)
    return reports


def finalize(
    output_root: Path,
    *,
    world_size: int,
    reference_driver: Path = DEFAULT_REFERENCE_DRIVER,
) -> dict[str, Any]:
    # Never leave an obsolete COMPLETE marker after a failed revalidation.
    (output_root / "control/COMPLETE").unlink(missing_ok=True)
    reference = load_reference_driver(reference_driver)
    manifest_rows, summary = load_manifest(output_root)
    reports = _load_rank_reports(output_root, world_size)
    current_fingerprint = runtime_fingerprint()
    for report in reports:
        if report.get("runtime_fingerprint") != current_fingerprint:
            raise ValueError("rank code changed; regenerate in a fresh output root")
        if report.get("release_eligible") is not True:
            raise ValueError("test-only/template recovery rank is not publishable")
        if report.get("manifest_sha256") != sha256_file(manifest_paths(output_root)[0]):
            raise ValueError("rank/source manifest fingerprint mismatch")
        if report.get("reference_driver_sha256") != sha256_file(reference_driver):
            raise ValueError("rank/reference driver fingerprint mismatch")
    expected_episodes = int(summary["episodes"])
    expected_blocks = int(summary["blocks"])
    expected_block_ids = {
        int(step["global_block_index"])
        for manifest in manifest_rows
        for step in manifest["segments"]
    }
    if expected_block_ids != set(range(expected_blocks)):
        raise ValueError("manifest global block IDs are not contiguous and unique")
    if sum(int(row["episodes"]) for row in reports) != expected_episodes:
        raise ValueError("rank episode totals do not match manifest")
    if sum(int(row["blocks"]) for row in reports) != expected_blocks:
        raise ValueError("rank block totals do not match manifest")

    vlm_rows: list[dict[str, Any]] = []
    vla_rows: list[dict[str, Any]] = []
    source_by_episode: dict[int, dict[str, Any]] = {}
    annotation_by_key: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        paths = shard_paths(output_root, rank)
        for row in jsonl_rows(paths["source"]):
            episode_index = int(row["global_episode_index"])
            if episode_index in source_by_episode:
                raise ValueError(f"duplicate source episode index {episode_index}")
            source_by_episode[episode_index] = row
        for row in jsonl_rows(paths["annotations"]):
            key = annotation_key(row)
            if key in annotation_by_key:
                raise ValueError(f"duplicate annotation key {key}")
            if not candidate_used_hidden_reasoning(row):
                raise ValueError(f"annotation {key} lacks substantive hidden reasoning")
            annotation_by_key[key] = row
        vlm_rows.extend(jsonl_rows(paths["vlm"]))
        vla_rows.extend(jsonl_rows(paths["vla"]))
    if set(source_by_episode) != set(range(expected_episodes)):
        raise ValueError("source episode indices are not contiguous and complete")
    if set(annotation_by_key) != expected_block_ids:
        missing = sorted(expected_block_ids - set(annotation_by_key))
        unexpected = sorted(set(annotation_by_key) - expected_block_ids)
        raise ValueError(
            "annotation global block coverage mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]}"
        )

    expected_targets: dict[tuple[int, int], dict[str, Any]] = {}
    expected_context: dict[tuple[int, int], dict[str, Any]] = {}
    diversity = DiversityAudit()
    for episode_index in range(expected_episodes):
        record = source_by_episode[episode_index]
        validate_segments(record["source_steps"])
        manifest_steps = manifest_rows[episode_index]["segments"]
        if len(record["source_steps"]) != len(manifest_steps):
            raise ValueError("source/manifest segment count mismatch")
        states, actions, _timestamps, _extras = load_native_endpose(
            Path(str(record["source_paths"]["recovery_hdf5"]))
        )
        if len(states) != int(record["all_frames"]):
            raise ValueError(
                f"episode {episode_index}: native end-pose frame count changed"
            )
        for segment_index, step in enumerate(record["source_steps"]):
            manifest_step = manifest_steps[segment_index]
            for field in (
                "start_frame",
                "end_frame",
                "left_action",
                "right_action",
                "global_block_index",
            ):
                if step[field] != manifest_step[field]:
                    raise ValueError(f"source/manifest mismatch: {field}")
            if int(step["source_segment_id"]) != segment_index:
                raise ValueError(
                    f"episode {episode_index}: source segment IDs are not contiguous"
                )
            block_index = step_annotation_key(step)
            annotation = annotation_by_key[block_index]
            checked = annotation_quality(annotation["candidate"], step)
            if any(annotation.get(k) != v for k, v in checked.items()):
                raise ValueError(
                    f"annotation {block_index}: recovery quality metadata mismatch"
                )
            if annotation["candidate"]["sub_task"] != step["source_sub_task"]:
                raise ValueError(
                    f"annotation {block_index}: sub_task differs from frozen action"
                )
            if step["source_sub_task"] != _generic_subtask(
                step["left_action"], step["right_action"]
            ):
                raise ValueError(
                    f"annotation {block_index}: noncanonical recovery sub_task"
                )
            if int(annotation["global_episode_index"]) != episode_index:
                raise ValueError(
                    f"block {block_index}: annotation episode index mismatch"
                )
            if str(annotation["episode_id"]) != str(record["episode_id"]):
                raise ValueError(
                    f"block {block_index}: annotation source episode mismatch"
                )
            if int(annotation["segment_id"]) != segment_index:
                raise ValueError(
                    f"block {block_index}: annotation segment index mismatch"
                )
            start = int(step["start_frame"])
            end = int(step["end_frame"])
            for arm in ("left", "right"):
                actual = _primitive_for_arm(
                    states, actions, start, end, arm, SegmentationConfig()
                )
                if actual != step[f"{arm}_action"]:
                    raise ValueError(
                        f"block {block_index}: {arm} primitive differs from native motion"
                    )
            expected_guide_action = _round_vector(actions[end] - states[start])
            if list(step["guide_action"]) != expected_guide_action:
                raise ValueError(
                    f"block {block_index}: source guide_action differs from native end-pose"
                )
            expected_chunks = int(math.ceil((end - start + 1) / CHUNK_SIZE))
            if int(step["num_chunks"]) != expected_chunks:
                raise ValueError(
                    f"block {block_index}: source num_chunks differs from frame span"
                )
            expected_frozen_action = {
                "left_action": str(step["left_action"]),
                "right_action": str(step["right_action"]),
                "primary_action_verb": str(step["primary_action_verb"]),
                "guide_action": expected_guide_action,
                "num_chunks": expected_chunks,
            }
            if annotation.get("frozen_action") != expected_frozen_action:
                raise ValueError(
                    f"block {block_index}: annotation frozen action differs "
                    "from native recovery target"
                )
            current_frame = int(annotation["current_frame"])
            if current_frame != start:
                raise ValueError(
                    f"block {block_index}: annotation current frame mismatch"
                )
            for image in annotation.get("visible_images", []):
                if int(image["frame"]) > current_frame:
                    raise ValueError(
                        f"block {block_index}: annotation exposes a future image"
                    )
                image_path = output_root / str(image["path"])
                if not image_path.is_file() or image_path.stat().st_size == 0:
                    raise FileNotFoundError(image_path)
            previous_step = (
                record["source_steps"][segment_index - 1] if segment_index else None
            )
            if annotation.get("visible_images") != _image_rows(
                output_root, record, step, previous_step
            ):
                raise ValueError(
                    f"block {block_index}: stale history/current image selection"
                )
            expected_targets[(episode_index, segment_index)] = {
                "guide_action": expected_guide_action,
                "primary_action_verb": str(step["primary_action_verb"]),
                "sub_task": str(step["source_sub_task"]),
                "num_chunks": expected_chunks,
            }
            expected_context[(episode_index, segment_index)] = {
                "current_state": states[start].tolist(),
                "guide_action_target_state": actions[end].tolist(),
                "segment_start_frame": start,
                "segment_end_frame": end,
                "previous_boundary_frame": (
                    int(record["source_steps"][segment_index - 1]["start_frame"])
                    if segment_index
                    else None
                ),
                "previous_boundary_state": (
                    states[
                        int(record["source_steps"][segment_index - 1]["start_frame"])
                    ].tolist()
                    if segment_index
                    else None
                ),
                "previous_goal": expected_targets.get(
                    (episode_index, segment_index - 1)
                ),
            }
    vlm_rows.sort(
        key=lambda row: (int(row["episode_index"]), int(row["segment_index"]))
    )
    vla_rows.sort(key=lambda row: int(row["global_episode_index"]))
    if len(vlm_rows) != expected_blocks or len(vla_rows) != expected_episodes:
        raise ValueError("merged VLM/VLA counts differ from manifest")
    seen: set[tuple[int, int]] = set()
    for row in vlm_rows:
        reference.validate_embodied_cot_row(row)
        validate_planner_row(row)
        key = (int(row["episode_index"]), int(row["segment_index"]))
        if key in seen:
            raise ValueError(f"duplicate VLM key {key}")
        seen.add(key)
        manifest = manifest_rows[key[0]]
        annotation = annotation_by_key[int(manifest["global_block_start"]) + key[1]]
        reasoning, answer = reference._split_embodied_assistant(
            str(row["messages"][2]["content"])
        )
        if reasoning != str(annotation["candidate"]["reasoning"]):
            raise ValueError(f"{key}: assistant reasoning differs from annotation")
        if answer != row["target"]:
            raise ValueError(f"{key}: assistant target mismatch")
        if row["target"] != expected_targets[key]:
            raise ValueError(f"{key}: VLM target differs from native recovery target")
        for field, expected in expected_context[key].items():
            if row[field] != expected:
                raise ValueError(f"{key}: stale/misaligned recovery context {field}")
        expected_images = [
            str(output_root / image["path"]) for image in annotation["visible_images"]
        ]
        if row["images"] != expected_images:
            raise ValueError(f"{key}: planner/teacher images differ")
        diversity.add(reasoning, row["target"], int(annotation["global_block_index"]))
        if any(
            _validate_critic(annotation[name])["verdict"] != "PASS"
            for name in ("critic_1", "critic_2")
        ):
            raise ValueError(f"{key}: critic did not pass")
    if seen != set(expected_targets):
        raise ValueError("VLM episode/segment coverage differs from source")
    vla_sample_ids: set[str] = set()
    vla_source_block_ids: set[int] = set()
    for expected_index, episode in enumerate(vla_rows):
        if int(episode["global_episode_index"]) != expected_index:
            raise ValueError("VLA episode indices are not contiguous")
        if FORBIDDEN_VLA_KEYS & set(_walk_keys(episode)):
            raise ValueError(f"VLA episode {expected_index} contains CoT")
        source_record = source_by_episode[expected_index]
        if str(episode["source_episode_id"]) != str(source_record["episode_id"]):
            raise ValueError(f"VLA episode {expected_index}: source identity mismatch")
        if len(episode["segments"]) != len(source_record["source_steps"]):
            raise ValueError(f"VLA episode {expected_index}: segment count mismatch")
        for segment_index, segment in enumerate(episode["segments"]):
            source_step = source_record["source_steps"][segment_index]
            for field in ("start_frame", "end_frame", "left_action", "right_action"):
                if segment[field] != source_step[field]:
                    raise ValueError(f"VLA source span/action mismatch: {field}")
            if int(segment["source_segment_id"]) != segment_index:
                raise ValueError(
                    f"VLA episode {expected_index}: segment IDs are not contiguous"
                )
            segment["sample_id"] = vla_sample_id(source_record, source_step)
            sample_id = str(segment["sample_id"])
            if sample_id in vla_sample_ids:
                raise ValueError(f"duplicate VLA sample_id {sample_id}")
            vla_sample_ids.add(sample_id)
            source_block = int(segment["source_global_block_index"])
            if source_block != step_annotation_key(source_step):
                raise ValueError(
                    f"VLA episode {expected_index} segment {segment_index}: "
                    "source block identity mismatch"
                )
            if source_block in vla_source_block_ids:
                raise ValueError(
                    f"duplicate VLA source_global_block_index {source_block}"
                )
            vla_source_block_ids.add(source_block)
            expected_target = expected_targets[(expected_index, segment_index)]
            actual_target = {
                key: segment[key]
                for key in (
                    "guide_action",
                    "primary_action_verb",
                    "sub_task",
                    "num_chunks",
                )
            }
            if actual_target != expected_target:
                raise ValueError(
                    f"VLA episode {expected_index} segment {segment_index}: "
                    "target differs from native recovery target"
                )
    if vla_source_block_ids != expected_block_ids:
        raise ValueError("VLA source global block coverage differs from manifest")
    quality_report = diversity.report()
    quality_report.update(
        {
            "nonterminal_both_still": 0,
            "short_adjacent_identical_action_boundaries": 0,
            "short_distinct_action_segments": sum(
                int(s["end_frame"]) - int(s["start_frame"]) + 1 < CHUNK_SIZE
                for record in source_by_episode.values()
                for s in record["source_steps"]
            ),
            "system_prompt_matches_reference_contract": True,
            "all_cot_rows_end_with_exact_action_sentence": True,
            "quality_policy": QUALITY_POLICY,
        }
    )
    vlm_path = output_root / "data/vlm/train.jsonl"
    vla_path = output_root / "data/vla/train.json"
    atomic_write_jsonl(vlm_path, vlm_rows)
    vla_payload: dict[str, Any] = {}
    for index, episode in enumerate(vla_rows):
        value = copy.deepcopy(episode)
        value["source_training_episode_index"] = int(value["training_episode_index"])
        value["training_episode_index"] = index
        for segment in value["segments"]:
            segment["training_episode_index"] = index
        vla_payload[str(index)] = value
    atomic_write_json(vla_path, vla_payload)
    action_path = output_root / "data/action_segmentation/train.jsonl"
    atomic_write_jsonl(action_path, manifest_rows)
    atomic_write_json(output_root / "COT_QUALITY_AUDIT.json", quality_report)
    hashes = {
        "data/vlm/train.jsonl": sha256_file(vlm_path),
        "data/vla/train.json": sha256_file(vla_path),
        "metadata/source_manifest.jsonl": sha256_file(manifest_paths(output_root)[0]),
        "data/action_segmentation/train.jsonl": sha256_file(action_path),
        "COT_QUALITY_AUDIT.json": sha256_file(output_root / "COT_QUALITY_AUDIT.json"),
    }
    critic_counts = Counter(
        annotation[name]["verdict"]
        for annotation in annotation_by_key.values()
        for name in ("critic_1", "critic_2")
    )
    inference_endpoints = non_loopback_endpoints(
        str(endpoint)
        for rank_report in reports
        for endpoint in rank_report.get("endpoints", [])
    )
    hidden_candidate_count = sum(
        candidate_used_hidden_reasoning(annotation)
        for annotation in annotation_by_key.values()
    )
    if hidden_candidate_count != expected_blocks:
        raise ValueError(
            "not every candidate annotation was generated with hidden "
            f"reasoning: {hidden_candidate_count}/{expected_blocks}"
        )
    report = {
        "schema_version": "robotwin_recovery_v28_final_report_v1",
        "code_version": CODE_VERSION,
        "created_at_utc": utc_now(),
        "passed": True,
        "source_dataset": "RoboTwin-Recovery",
        "source_root": summary["source_root"],
        "world_size": world_size,
        "inference_endpoint_count": len(inference_endpoints),
        "inference_endpoints": inference_endpoints,
        "gpu_topology": (
            f"{len(inference_endpoints)} TP8 endpoints = "
            f"{len(inference_endpoints) * 8} GPUs"
        ),
        "episodes": expected_episodes,
        "vlm_rows": len(vlm_rows),
        "vla_episodes": len(vla_payload),
        "vla_segments": sum(len(value["segments"]) for value in vla_payload.values()),
        "annotation_count": len(annotation_by_key),
        "hidden_reasoning_candidate_count": hidden_candidate_count,
        "critic_verdict_counts": dict(critic_counts),
        "state_layout": list(STATE_LAYOUT),
        "vlm_format_version": VLM_FORMAT_VERSION,
        "hashes": hashes,
        "artifact_hashes": hashes,
        "counts": {
            "annotations": len(annotation_by_key),
            "vlm_rows": len(vlm_rows),
            "vla_episodes": len(vla_payload),
            "vla_segments": sum(
                len(value["segments"]) for value in vla_payload.values()
            ),
        },
        "outputs": {
            "action_segmentation": str(action_path),
            "source_manifest": str(manifest_paths(output_root)[0]),
            "vlm_train": str(vlm_path),
            "vla_train": str(vla_path),
        },
        "quality_contract": {
            **quality_report,
            "one_row_per_source_block": True,
            "embodied_row_schema_preserved": True,
            "assistant_final_json_equals_target": True,
            "assistant_reasoning_matches_saved_annotation": True,
            "dual_model_critics_per_block": True,
            "all_candidates_used_substantive_hidden_reasoning": True,
            "vlm_target_matches_recovery_endpose": True,
            "vla_target_matches_recovery_endpose": True,
            "vla_contains_cot": False,
            "vla_sample_ids_unique": True,
        },
        "rank_reports": reports,
    }
    audit = {
        "schema_version": "robotwin_recovery_v28_embodied_alignment_audit_v1",
        "passed": True,
        "all_v28_critics_passed": critic_counts
        == Counter({"PASS": expected_blocks * 2}),
        "all_candidates_used_hidden_reasoning": (
            hidden_candidate_count == expected_blocks
        ),
        "assistant_reasoning_matches_saved_annotation": True,
        "artifact_hashes_verified": hashes,
        "annotation_global_block_ids_match_manifest": True,
        "unique_source_block_ids": len(annotation_by_key),
        "vla_contains_cot": False,
        "vla_unique_sample_ids": len(vla_sample_ids),
        "vla_unique_source_block_ids": len(vla_source_block_ids),
        "vla_episodes": len(vla_payload),
        "vla_segments": sum(len(value["segments"]) for value in vla_payload.values()),
        "vla_target_matches_recovery_endpose": True,
        "vlm_rows": len(vlm_rows),
        "vlm_target_matches_recovery_endpose": True,
        "vlm_unique_episode_segment_keys": len(seen),
    }
    atomic_write_json(output_root / "FINAL_REPORT.json", report)
    atomic_write_json(output_root / "FINAL_EMBODIED_ALIGNMENT_AUDIT.json", audit)
    atomic_write_text(
        output_root / "README.md",
        "# RoboTwin recovery CoT dataset\n\n"
        "Recommended training files:\n\n"
        "* `data/vlm/train.jsonl`: recovery VLM samples with causal pre-action "
        "CoT in `<think>...</think>`.\n"
        "* `data/vla/train.json`: aligned recovery action-only episodes; no CoT "
        "fields are present.\n\n"
        "The VLM rows use the same 54-field Embodied planner contract as "
        "`robotwin_cot`, with 16-D end-effector targets reconstructed from the "
        "audited native recovery HDF5 trajectories.\n",
    )
    atomic_write_text(output_root / "control/COMPLETE", utc_now() + "\n")
    return report


def status(output_root: Path, *, world_size: int) -> dict[str, Any]:
    _manifest_path, summary_path = manifest_paths(output_root)
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.is_file()
        else {}
    )
    ranks = []
    valid_block_ids: set[int] = set()
    raw_annotation_rows = 0
    for rank in range(world_size):
        paths = shard_paths(output_root, rank)
        prepare_report = (
            json.loads(paths["prepare_report"].read_text(encoding="utf-8"))
            if paths["prepare_report"].is_file()
            else {}
        )
        valid_annotations: set[int] = set()
        rank_raw_rows = 0
        if paths["annotations"].is_file():
            with paths["annotations"].open(encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rank_raw_rows += 1
                    if (
                        candidate_used_hidden_reasoning(row)
                        and row.get("quality_policy") == QUALITY_POLICY
                    ):
                        valid_annotations.add(annotation_key(row))
        valid_block_ids.update(valid_annotations)
        raw_annotation_rows += rank_raw_rows
        ranks.append(
            {
                "rank": rank,
                "prepared_episodes": int(prepare_report.get("episodes", 0)),
                "prepared_blocks": int(prepare_report.get("blocks", 0)),
                "annotations": len(valid_annotations),
                "raw_annotation_rows": rank_raw_rows,
                "done": paths["done"].is_file(),
                "failed": paths["failures"].is_file(),
            }
        )
    complete_episodes = 0
    partial_episodes = 0
    unstarted_episodes = 0
    manifest_path, _summary_path = manifest_paths(output_root)
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                start = int(record["global_block_start"])
                count = int(record["segment_count"])
                present = sum(
                    block in valid_block_ids for block in range(start, start + count)
                )
                if count and present == count:
                    complete_episodes += 1
                elif present:
                    partial_episodes += 1
                else:
                    unstarted_episodes += 1
    manifest_blocks = summary.get("blocks")
    return {
        "code_version": CODE_VERSION,
        "manifest_episodes": summary.get("episodes"),
        "manifest_blocks": manifest_blocks,
        "prepared_episodes": sum(row["prepared_episodes"] for row in ranks),
        "prepared_blocks": sum(row["prepared_blocks"] for row in ranks),
        "annotations": len(valid_block_ids),
        "raw_annotation_rows": raw_annotation_rows,
        "remaining_blocks": (
            int(manifest_blocks) - len(valid_block_ids)
            if manifest_blocks is not None
            else None
        ),
        "complete_episodes": complete_episodes,
        "partial_episodes": partial_episodes,
        "unstarted_episodes": unstarted_episodes,
        "completed_ranks": sum(bool(row["done"]) for row in ranks),
        "failed_ranks": [row["rank"] for row in ranks if row["failed"]],
        "complete": (output_root / "control/COMPLETE").is_file(),
        "ranks": ranks,
    }
