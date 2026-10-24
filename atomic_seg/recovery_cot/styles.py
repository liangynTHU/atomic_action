"""Qwen regeneration of stylistically diverse RoboTwin CoT.

The source release remains read-only.  Each source row keeps its system/user
messages, images, target, and trajectory metadata exactly.  Only the reasoning
inside the assistant ``<think>`` block is regenerated from the original
planner context, a fixed target, and the previously multimodally audited
reasoning as a factual anchor.  The rewrite model does not reinterpret images.
"""

from __future__ import annotations

import concurrent.futures
import copy
import difflib
import hashlib
import json
import os
import re
import shutil
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from atomic_seg.common.artifacts import compact_json, sha256_file, utc_now
from atomic_seg.common.artifacts import iter_jsonl as jsonl_rows
from atomic_seg.common.cot_format import assistant_with_reasoning, split_assistant
from atomic_seg.common.cot_format import word_count as _word_count
from atomic_seg.common.model_client import normalize_endpoint, request_model
from atomic_seg.cot_data.validation import _contradictory_action_issues
from .quality import DiversityAudit, complete_reasoning, reasoning_body, validate_body
from .planner_contract import validate_planner_row
from .policy import QUALITY_POLICY, validate_segments

SCHEMA_VERSION = "robotwin_recovery_cot_style5_generation_v1"
ANNOTATION_SCHEMA_VERSION = "robotwin_recovery_cot_style5_annotation_v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_INPUT_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/datasets/"
    "robotwin_recovery_cot_still_merged"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/datasets/"
    "robotwin_recovery_cot_still_merged_cot5"
)
STYLE_SPECS = (
    {
        "name": "evidence_first",
        "brief": (
            "Lead with the most diagnostic camera/state observations. Connect "
            "those facts to whether the prior command completed, then justify "
            "the fixed next atomic action. Keep the prose compact and forensic."
        ),
    },
    {
        "name": "temporal_contrast",
        "brief": (
            "Organize the reasoning as a before-versus-now comparison. Explain "
            "what changed across the history boundary, what remained unchanged, "
            "and how that temporal contrast supports the fixed next action."
        ),
    },
    {
        "name": "goal_decomposition",
        "brief": (
            "Frame the reasoning around the task's current subgoal: identify the "
            "completed prerequisite, the unresolved immediate requirement, and "
            "why the fixed action is the next necessary atomic step."
        ),
    },
    {
        "name": "alternative_elimination",
        "brief": (
            "Briefly distinguish the fixed next action from plausible alternatives "
            "such as repeating, switching arms, opening/closing, or holding still. "
            "Reject alternatives only using visible/state evidence, then commit."
        ),
    },
    {
        "name": "uncertainty_calibrated",
        "brief": (
            "Separate direct observations from physical inferences. State any "
            "occlusion or contact uncertainty conservatively, while still giving "
            "a clear causal justification for the fixed next atomic action."
        ),
    },
)


def source_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["episode_index"]), int(row["segment_index"])


def style_id_for_row(row_index: int) -> int:
    if row_index < 0:
        raise ValueError("row_index must be non-negative")
    return row_index % len(STYLE_SPECS)


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def lexical_similarity(left: str, right: str) -> float:
    left_words = _normalized_words(left)
    right_words = _normalized_words(right)
    return difflib.SequenceMatcher(None, left_words, right_words).ratio()


def _ngrams(words: Sequence[str], size: int) -> set[tuple[str, ...]]:
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {
        tuple(words[index : index + size]) for index in range(len(words) - size + 1)
    }


def ngram_overlap(left: str, right: str, size: int = 4) -> float:
    left_grams = _ngrams(_normalized_words(left), size)
    right_grams = _ngrams(_normalized_words(right), size)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / max(
        1, min(len(left_grams), len(right_grams))
    )


def _reasoning_json(value: Mapping[str, Any]) -> str:
    if set(value) != {"reasoning"}:
        raise ValueError("generation response must contain only reasoning")
    reasoning = re.sub(r"\s+", " ", str(value["reasoning"])).strip()
    words = _word_count(reasoning)
    if not 70 <= words <= 190:
        raise ValueError(f"reasoning length is {words} words, expected 70-190")
    lower = reasoning.lower()
    forbidden = (
        "guide_action",
        "ground truth",
        "future frame",
        "future image",
        "segment endpoint",
        "style brief",
        "audited reasoning",
        "source reasoning",
        "frozen target",
    )
    if any(token in lower for token in forbidden):
        raise ValueError("reasoning exposes internal supervision language")
    if reasoning.startswith(("Evidence:", "Analysis:", "Reasoning:", "Style:")):
        raise ValueError("reasoning must be natural prose without a heading")
    return reasoning


def _image_roles(row: Mapping[str, Any]) -> list[str]:
    if len(row["images"]) == 3:
        return ["current_head", "current_left_wrist", "current_right_wrist"]
    if len(row["images"]) == 6:
        return [
            "previous_boundary_head",
            "previous_boundary_left_wrist",
            "previous_boundary_right_wrist",
            "current_head",
            "current_left_wrist",
            "current_right_wrist",
        ]
    raise ValueError("source row must contain three or six images")


def _generation_prompt(
    row: Mapping[str, Any],
    *,
    original_reasoning: str,
    style_id: int,
    revision_issues: Sequence[str],
) -> str:
    spec = STYLE_SPECS[style_id]
    parts = [
        "Original planner input (keep its meaning and chronology):",
        str(row["messages"][1]["content"]).replace("<image>", ""),
        "Fixed next-action target (authoritative; do not change it):",
        compact_json(
            {key: row["target"][key] for key in ("primary_action_verb", "sub_task")}
        ),
        "Previously audited factual reasoning anchor:",
        reasoning_body(original_reasoning, row["target"]),
        (
            "Hard robot-state convention: gripper values near 1 mean OPEN and "
            "values near 0 mean CLOSED. Never reverse this convention."
        ),
        f"Requested prose style: {spec['name']}",
        str(spec["brief"]),
        (
            "Write a fresh causal pre-action rationale grounded in the audited "
            "anchor and supplied state/history text. Preserve the anchor's supported physical facts and "
            "decision, but do not copy its sentence order, opening phrase, or "
            "surface wording. Assess only the already-executed previous command. "
            "The fixed next action has not happened yet, so do not claim its result. "
            "Name the fixed action's natural motion semantic explicitly (for example "
            "lift, turn, move, close, open, or hold still), but never print the "
            "internal guide_action field name or numeric target. "
            "Mention the active and stationary arms when relevant. Use one natural "
            "English paragraph of 70-190 words, no heading, no bullets, no markdown, "
            "and no internal dataset terminology. Return strict JSON with exactly "
            'one key: {"reasoning":"..."}.'
            " Do not write the final Therefore action sentence; the program adds it."
        ),
    ]
    if revision_issues:
        parts.extend(
            [
                "Revision requirements from the critic:",
                compact_json(list(revision_issues)),
            ]
        )
    return "\n".join(parts)


def _target_semantic_issues(
    candidate: str,
    target: Mapping[str, Any],
    style_id: int,
) -> list[str]:
    lower = candidate.lower()
    issues: list[str] = []
    issues.extend(_contradictory_action_issues(candidate, target))
    primary = str(target["primary_action_verb"]).lower()
    per_arm: dict[str, str] = {}
    for item in primary.split(","):
        arm, separator, action = item.strip().partition(":")
        if not separator or arm not in {"left", "right"}:
            raise ValueError(f"invalid primary_action_verb: {primary!r}")
        per_arm[arm] = action.strip()
    synonyms = {
        "move": (
            "move",
            "translate",
            "carry",
            "reposition",
            "shift",
            "transport",
            "raise",
            "lower",
            "lift",
        ),
        "turn": ("turn", "rotate", "reorient", "pivot", "twist"),
        "lift": ("lift", "raise", "upward", "elevate", "ascend"),
        "lower": ("lower", "descend", "downward"),
        "close_gripper": ("close", "grasp", "clamp", "secure", "grip"),
        "open_gripper": ("open", "release", "unclamp"),
        "still": ("still", "stationary", "hold", "remain", "park", "idle"),
    }
    for arm, action in per_arm.items():
        if arm not in lower:
            issues.append(f"Explicitly mention the {arm} arm.")
        action_terms = synonyms.get(action)
        if action_terms is None:
            raise ValueError(f"unsupported target action: {action}")
        if not any(term in lower for term in action_terms):
            issues.append(f"Explicitly express the target semantic {arm}:{action}.")
    style_terms = (
        ("view", "state", "shows", "evidence", "confirm"),
        ("before", "previous", "now", "current", "across", "while"),
        ("prerequisite", "requirement", "subgoal", "remaining", "next"),
        ("rather", "repeat", "switch", "opening", "closing", "unnecessary"),
        (
            "uncertain",
            "cannot confirm",
            "not confirm",
            "suggest",
            "likely",
            "appears",
            "inference",
            "direct observation",
            "visually clear",
            "evidence is clear",
            "unambiguous",
            "ambiguous",
            "occluded",
        ),
    )
    required_matches = 2 if style_id in {1, 3, 4} else 1
    if sum(term in lower for term in style_terms[style_id]) < required_matches:
        issues.append(
            f"Make the requested {STYLE_SPECS[style_id]['name']} structure more explicit."
        )
    return issues


def _request_with_retry(
    endpoint_values: Sequence[str],
    *,
    endpoint_offset: int,
    model: str,
    system: str,
    prompt: str,
    image_paths: Sequence[Path],
    image_roles: Sequence[str],
    timeout: float,
    max_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors: list[str] = []
    for request_attempt in range(attempts):
        endpoint_value = endpoint_values[
            (endpoint_offset + request_attempt) % len(endpoint_values)
        ]
        try:
            return request_model(
                normalize_endpoint(endpoint_value),
                model=model,
                system=system,
                prompt=prompt,
                image_paths=image_paths,
                image_roles=image_roles,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed + request_attempt * 1000003,
                enable_thinking=enable_thinking,
            )
        except Exception as exc:  # noqa: BLE001 - preserve remote retry context
            errors.append(f"{endpoint_value}: {type(exc).__name__}: {exc}")
    raise RuntimeError("all endpoint attempts failed: " + " | ".join(errors))


def generate_annotation(
    row_index: int,
    row: Mapping[str, Any],
    *,
    endpoint_values: Sequence[str],
    model: str,
    timeout: float,
    generation_max_tokens: int,
    max_rounds: int,
    seed_salt: int = 0,
    initial_revision_issues: Sequence[str] = (),
) -> dict[str, Any]:
    validate_planner_row(row)
    messages = row["messages"]
    if [message.get("role") for message in messages] != [
        "system",
        "user",
        "assistant",
    ]:
        raise ValueError("source messages are not system/user/assistant")
    original_reasoning, answer_text = split_assistant(str(messages[2]["content"]))
    if json.loads(answer_text) != row["target"]:
        raise ValueError("source assistant target differs from top-level target")
    image_paths = [Path(str(value)) for value in row["images"]]
    for image_path in image_paths:
        if not image_path.is_file() or image_path.stat().st_size == 0:
            raise FileNotFoundError(image_path)
    _image_roles(row)  # Validate source roles; the rewrite itself is text-only.
    style_id = style_id_for_row(row_index)
    revision_issues = list(initial_revision_issues)
    round_records: list[dict[str, Any]] = []
    for round_index in range(max_rounds):
        generation_value, generation_metadata = _request_with_retry(
            endpoint_values,
            endpoint_offset=row_index + round_index,
            model=model,
            system=(
                "You write diverse, evidence-grounded pre-action reasoning for a "
                "RoboTwin high-level planner. The target action is fixed supervision. "
                "Use the images and state history, avoid future leakage, and output "
                "strict JSON."
            ),
            prompt=_generation_prompt(
                row,
                original_reasoning=original_reasoning,
                style_id=style_id,
                revision_issues=revision_issues,
            ),
            image_paths=(),
            image_roles=(),
            timeout=timeout,
            max_tokens=generation_max_tokens,
            temperature=0.65,
            seed=2026092700 + seed_salt + row_index * 17 + round_index,
            # The source CoT is already an audited factual plan.  This pass is a
            # constrained multimodal rewrite, so hidden deliberation adds large
            # latency without adding new supervision.
            enable_thinking=False,
        )
        try:
            candidate = _reasoning_json(generation_value)
        except Exception as exc:  # noqa: BLE001
            revision_issues = [str(exc)]
            round_records.append(
                {
                    "round": round_index + 1,
                    "generation": generation_metadata,
                    "static_validation": "REVISE",
                    "issues": revision_issues,
                }
            )
            continue
        original_body = reasoning_body(original_reasoning, row["target"])
        similarity = lexical_similarity(original_body, candidate)
        overlap = ngram_overlap(original_body, candidate)
        static_issues: list[str] = []
        if similarity > 0.82:
            static_issues.append(
                f"Restructure more substantially; lexical similarity is {similarity:.3f}."
            )
        if overlap > 0.58:
            static_issues.append(
                f"Avoid copying long phrases; four-gram overlap is {overlap:.3f}."
            )
        if static_issues:
            revision_issues = static_issues
            round_records.append(
                {
                    "round": round_index + 1,
                    "generation": generation_metadata,
                    "static_validation": "REVISE",
                    "similarity": round(similarity, 6),
                    "fourgram_overlap": round(overlap, 6),
                    "issues": revision_issues,
                }
            )
            continue
        semantic_issues = _target_semantic_issues(candidate, row["target"], style_id)
        try:
            validate_body(candidate, row["target"], style_id)
        except ValueError as exc:
            semantic_issues.append(str(exc))
        round_record = {
            "round": round_index + 1,
            "generation": generation_metadata,
            "static_validation": "REVISE" if semantic_issues else "PASS",
            "similarity": round(similarity, 6),
            "fourgram_overlap": round(overlap, 6),
            "issues": semantic_issues,
        }
        round_records.append(round_record)
        if not semantic_issues:
            return {
                "schema_version": ANNOTATION_SCHEMA_VERSION,
                "row_index": row_index,
                "episode_index": int(row["episode_index"]),
                "segment_index": int(row["segment_index"]),
                "style_id": style_id,
                "style_name": STYLE_SPECS[style_id]["name"],
                "model": model,
                "reasoning": candidate,
                "source_reasoning_sha256": hashlib.sha256(
                    original_reasoning.encode("utf-8")
                ).hexdigest(),
                "reasoning_sha256": hashlib.sha256(
                    candidate.encode("utf-8")
                ).hexdigest(),
                "word_count": _word_count(candidate),
                "lexical_similarity_to_source": round(similarity, 6),
                "fourgram_overlap_to_source": round(overlap, 6),
                "accepted_round": round_index + 1,
                "rounds": round_records,
            }
        revision_issues = semantic_issues
    raise RuntimeError(
        f"row {row_index} {source_key(row)} did not pass after {max_rounds} rounds: "
        + compact_json(round_records[-1] if round_records else {})
    )


def _read_endpoints(path: Path) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise ValueError(f"endpoint file is empty: {path}")
    return values


def _completed_annotations(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = list(jsonl_rows(path))
    for index, row in enumerate(output):
        if int(row["row_index"]) != index:
            raise ValueError("annotation row indices are not contiguous")
    return output


def generate_annotations(
    input_root: Path,
    output_root: Path,
    *,
    endpoint_file: Path,
    model: str = DEFAULT_MODEL,
    workers: int = 64,
    batch_size: int = 256,
    timeout: float = 600.0,
    generation_max_tokens: int = 4096,
    max_rounds: int = 4,
    job_attempts: int = 3,
    limit_rows: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0 or batch_size <= 0 or max_rounds <= 0 or job_attempts <= 0:
        raise ValueError(
            "workers, batch_size, max_rounds, and job_attempts must be positive"
        )
    input_vlm = input_root / "data/vlm/train.jsonl"
    if not input_vlm.is_file():
        raise FileNotFoundError(input_vlm)
    endpoints = _read_endpoints(endpoint_file)
    annotation_path = output_root / "work/cot_style_annotations.jsonl"
    annotation_path.parent.mkdir(parents=True, exist_ok=True)
    completed = _completed_annotations(annotation_path) if resume else []
    if annotation_path.exists() and not resume:
        raise FileExistsError(
            f"annotation file exists; use --resume or a new output root: {annotation_path}"
        )
    start_index = len(completed)
    source_stream = input_vlm.open("r", encoding="utf-8")
    try:
        for _ in range(start_index):
            if not source_stream.readline():
                raise ValueError("annotations exceed source VLM row count")
        rows_written = start_index
        with annotation_path.open("a", encoding="utf-8") as destination:
            while limit_rows is None or rows_written < limit_rows:
                batch: list[tuple[int, dict[str, Any]]] = []
                while len(batch) < batch_size:
                    if (
                        limit_rows is not None
                        and rows_written + len(batch) >= limit_rows
                    ):
                        break
                    line = source_stream.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("source VLM line is not an object")
                    batch.append((rows_written + len(batch), value))
                if not batch:
                    break
                results: dict[int, dict[str, Any]] = {}
                pending = {row_index: row for row_index, row in batch}
                last_errors: dict[int, str] = {}
                for job_attempt in range(job_attempts):
                    if not pending:
                        break
                    current_pending = pending
                    pending = {}
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(workers, len(current_pending))
                    ) as executor:
                        futures = {
                            executor.submit(
                                generate_annotation,
                                row_index,
                                row,
                                endpoint_values=endpoints,
                                model=model,
                                timeout=timeout,
                                generation_max_tokens=generation_max_tokens,
                                max_rounds=max_rounds,
                                seed_salt=job_attempt * 100000000,
                            ): (row_index, row)
                            for row_index, row in current_pending.items()
                        }
                        for future in concurrent.futures.as_completed(futures):
                            row_index, row = futures[future]
                            try:
                                results[row_index] = future.result()
                            except Exception as exc:  # noqa: BLE001
                                pending[row_index] = row
                                last_errors[row_index] = f"{type(exc).__name__}: {exc}"
                    if pending:
                        print(
                            compact_json(
                                {
                                    "event": "cot_style_retry",
                                    "job_attempt": job_attempt + 1,
                                    "failed_rows": len(pending),
                                    "first_failed": min(pending),
                                    "time": utc_now(),
                                }
                            ),
                            flush=True,
                        )
                if pending:
                    first_failed = min(pending)
                    raise RuntimeError(
                        f"{len(pending)} rows failed after {job_attempts} job "
                        f"attempts; first row {first_failed}: "
                        f"{last_errors[first_failed]}"
                    )
                for row_index, _row in batch:
                    destination.write(compact_json(results[row_index]) + "\n")
                destination.flush()
                os.fsync(destination.fileno())
                rows_written += len(batch)
                print(
                    compact_json(
                        {
                            "event": "cot_style_progress",
                            "rows": rows_written,
                            "last_key": source_key(batch[-1][1]),
                            "time": utc_now(),
                        }
                    ),
                    flush=True,
                )
    finally:
        source_stream.close()
    return {
        "annotations": rows_written,
        "annotation_path": str(annotation_path),
        "endpoint_count": len(endpoints),
    }


def _opening_prefix(reasoning: str, words: int = 8) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9']+", reasoning)[:words])


def refresh_repetitive_openings(
    input_root: Path,
    output_root: Path,
    *,
    endpoint_file: Path,
    model: str = DEFAULT_MODEL,
    workers: int = 128,
    batch_size: int = 512,
    prefix_words: int = 8,
    max_prefix_count: int = 50,
    timeout: float = 600.0,
    generation_max_tokens: int = 4096,
    max_rounds: int = 4,
    job_attempts: int = 5,
) -> dict[str, Any]:
    """Regenerate rows whose opening phrase is overrepresented."""

    if prefix_words <= 0 or max_prefix_count <= 0:
        raise ValueError("prefix_words and max_prefix_count must be positive")
    annotation_path = output_root / "work/cot_style_annotations.jsonl"
    annotations = list(jsonl_rows(annotation_path))
    prefix_counts = Counter(
        _opening_prefix(str(row["reasoning"]), prefix_words) for row in annotations
    )
    selected = {
        index
        for index, row in enumerate(annotations)
        if prefix_counts[_opening_prefix(str(row["reasoning"]), prefix_words)]
        > max_prefix_count
    }
    if not selected:
        return {
            "selected": 0,
            "updated": 0,
            "max_prefix_count": max(prefix_counts.values(), default=0),
        }
    source_rows: dict[int, dict[str, Any]] = {}
    source_path = input_root / "data/vlm/train.jsonl"
    for index, row in enumerate(jsonl_rows(source_path)):
        if index in selected:
            source_rows[index] = row
    if set(source_rows) != selected:
        raise ValueError("could not resolve every selected source row")
    endpoints = _read_endpoints(endpoint_file)
    opening_cues = (
        "the active gripper state",
        "the passive arm's unchanged state",
        "the object's current pose",
        "the most informative numeric state delta",
        "the task's arm constraint",
        "the observed outcome of the previous command",
        "agreement between camera and state evidence",
        "the immediate unresolved subgoal",
        "why interference from the other arm should be avoided",
        "the prerequisite that has just been completed",
        "the current planning boundary",
        "the causal transition needed next",
    )
    selected_order = sorted(selected)
    for offset in range(0, len(selected_order), batch_size):
        batch_indices = selected_order[offset : offset + batch_size]
        pending = set(batch_indices)
        last_errors: dict[int, str] = {}
        for job_attempt in range(job_attempts):
            if not pending:
                break
            current_pending = sorted(pending)
            pending = set()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(workers, len(current_pending))
            ) as executor:
                futures = {}
                for index in current_pending:
                    old_prefix = _opening_prefix(
                        str(annotations[index]["reasoning"]), prefix_words
                    )
                    cue = opening_cues[(index + job_attempt) % len(opening_cues)]
                    future = executor.submit(
                        generate_annotation,
                        index,
                        source_rows[index],
                        endpoint_values=endpoints,
                        model=model,
                        timeout=timeout,
                        generation_max_tokens=generation_max_tokens,
                        max_rounds=max_rounds,
                        seed_salt=500000000 + job_attempt * 100000000,
                        initial_revision_issues=(
                            f"Do not begin with or closely imitate this overused opening: {old_prefix!r}.",
                            f"Vary the opening syntax by beginning from {cue}.",
                        ),
                    )
                    futures[future] = index
                for future in concurrent.futures.as_completed(futures):
                    index = futures[future]
                    try:
                        candidate = future.result()
                        new_prefix = _opening_prefix(
                            str(candidate["reasoning"]), prefix_words
                        )
                        old_prefix = _opening_prefix(
                            str(annotations[index]["reasoning"]), prefix_words
                        )
                        if new_prefix == old_prefix:
                            raise ValueError("regeneration repeated the banned opening")
                        annotations[index] = candidate
                    except Exception as exc:  # noqa: BLE001
                        pending.add(index)
                        last_errors[index] = f"{type(exc).__name__}: {exc}"
            if pending:
                print(
                    compact_json(
                        {
                            "event": "opening_refresh_retry",
                            "batch_start": offset,
                            "job_attempt": job_attempt + 1,
                            "failed_rows": len(pending),
                            "first_failed": min(pending),
                            "time": utc_now(),
                        }
                    ),
                    flush=True,
                )
        if pending:
            first_failed = min(pending)
            raise RuntimeError(
                f"{len(pending)} opening refresh rows failed; first row "
                f"{first_failed}: {last_errors[first_failed]}"
            )
        print(
            compact_json(
                {
                    "event": "opening_refresh_progress",
                    "updated": min(offset + len(batch_indices), len(selected_order)),
                    "selected": len(selected_order),
                    "time": utc_now(),
                }
            ),
            flush=True,
        )
    temporary = annotation_path.with_name(
        f".{annotation_path.name}.refresh-{os.getpid()}"
    )
    with temporary.open("w", encoding="utf-8") as stream:
        for annotation in annotations:
            stream.write(compact_json(annotation) + "\n")
    backup = annotation_path.with_name(f"{annotation_path.name}.before_prefix_refresh")
    if not backup.exists():
        shutil.copy2(annotation_path, backup)
    os.replace(temporary, annotation_path)
    refreshed_counts = Counter(
        _opening_prefix(str(row["reasoning"]), prefix_words) for row in annotations
    )
    return {
        "selected": len(selected_order),
        "updated": len(selected_order),
        "prefix_words": prefix_words,
        "threshold": max_prefix_count,
        "before_max_prefix_count": max(prefix_counts.values()),
        "after_max_prefix_count": max(refreshed_counts.values()),
        "after_unique_prefixes": len(refreshed_counts),
    }


def _copy_without_assistant_reasoning(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(row)
    _reasoning, answer_text = split_assistant(str(value["messages"][2]["content"]))
    value["messages"][2]["content"] = assistant_with_reasoning(
        "<REASONING>", answer_text
    )
    return value


def finalize_dataset(
    input_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if input_root.resolve() == output_root.resolve():
        raise ValueError("style rewrite requires a separate output root")
    if (output_root / "control/COMPLETE").exists():
        raise ValueError(
            "style experiments cannot overwrite a completed formal release"
        )
    source_audit = json.loads((input_root / "COT_QUALITY_AUDIT.json").read_text())
    if (
        source_audit.get("quality_policy") != QUALITY_POLICY
        or source_audit.get("passed") is not True
    ):
        raise ValueError(
            "old recovery source: regenerate through the native quality pipeline"
        )
    input_vlm = input_root / "data/vlm/train.jsonl"
    input_vla = input_root / "data/vla/train.json"
    annotation_path = output_root / "work/cot_style_annotations.jsonl"
    output_vlm = output_root / "data/vlm/train.jsonl"
    output_vla = output_root / "data/vla/train.json"
    report_path = output_root / "COT_STYLE_DIVERSIFICATION_REPORT.json"
    readme_path = output_root / "README.md"
    for path in (output_vlm, output_vla, report_path, readme_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite final output: {path}")
    annotations = jsonl_rows(annotation_path)
    output_vlm.parent.mkdir(parents=True, exist_ok=True)
    output_vla.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_vlm.with_name(f".{output_vlm.name}.tmp-{os.getpid()}")
    style_counts: Counter[str] = Counter()
    accepted_rounds: Counter[int] = Counter()
    word_counts: list[int] = []
    similarities: list[float] = []
    overlaps: list[float] = []
    reasoning_hashes: set[str] = set()
    generation_endpoints: set[str] = set()
    hidden_reasoning_generations = 0
    rows = 0
    diversity = DiversityAudit()
    annotation_iter = iter(annotations)
    try:
        with (
            input_vlm.open("r", encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as destination,
        ):
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                source_row = json.loads(line)
                validate_planner_row(source_row)
                try:
                    annotation = next(annotation_iter)
                except StopIteration as exc:
                    raise ValueError("annotations are incomplete") from exc
                if int(annotation["row_index"]) != rows:
                    raise ValueError("annotation index mismatch")
                if source_key(source_row) != (
                    int(annotation["episode_index"]),
                    int(annotation["segment_index"]),
                ):
                    raise ValueError("annotation source key mismatch")
                original_reasoning, answer_text = split_assistant(
                    str(source_row["messages"][2]["content"])
                )
                if hashlib.sha256(
                    original_reasoning.encode("utf-8")
                ).hexdigest() != str(annotation["source_reasoning_sha256"]):
                    raise ValueError("source reasoning changed after annotation")
                output_row = copy.deepcopy(source_row)
                body = reasoning_body(
                    str(annotation["reasoning"]), source_row["target"], required=False
                )
                if int(annotation["style_id"]) != rows % len(STYLE_SPECS):
                    raise ValueError("style annotation allocation mismatch")
                if int(annotation["accepted_round"]) < 1 or annotation.get("fallback"):
                    raise ValueError("unaccepted/fallback recovery style annotation")
                if (
                    hashlib.sha256(str(annotation["reasoning"]).encode()).hexdigest()
                    != annotation["reasoning_sha256"]
                ):
                    raise ValueError("style annotation reasoning hash mismatch")
                reasoning = complete_reasoning(body, source_row["target"])
                diversity.add(reasoning, source_row["target"], rows)
                output_row["messages"][2]["content"] = assistant_with_reasoning(
                    reasoning, answer_text
                )
                if _copy_without_assistant_reasoning(
                    source_row
                ) != _copy_without_assistant_reasoning(output_row):
                    raise ValueError("a non-reasoning VLM field changed")
                destination.write(compact_json(output_row) + "\n")
                style_counts[str(annotation["style_name"])] += 1
                accepted_rounds[int(annotation["accepted_round"])] += 1
                word_counts.append(int(annotation["word_count"]))
                similarities.append(float(annotation["lexical_similarity_to_source"]))
                overlaps.append(float(annotation["fourgram_overlap_to_source"]))
                reasoning_hash = str(annotation["reasoning_sha256"])
                reasoning_hashes.add(reasoning_hash)
                for round_record in annotation["rounds"]:
                    generation = round_record.get("generation", {})
                    endpoint = generation.get("endpoint")
                    if endpoint:
                        generation_endpoints.add(str(endpoint))
                    if generation.get("hidden_reasoning_present"):
                        hidden_reasoning_generations += 1
                rows += 1
            try:
                next(annotation_iter)
            except StopIteration:
                pass
            else:
                raise ValueError("annotations exceed source row count")
        quality_report = diversity.report()
        vla_data = json.loads(input_vla.read_text())
        for episode in vla_data.values():
            validate_segments(episode["segments"])
        os.replace(temporary, output_vlm)
    finally:
        if temporary.exists():
            temporary.unlink()

    shutil.copy2(input_vla, output_vla)
    for name in (
        "metadata/source_manifest.jsonl",
        "data/action_segmentation/train.jsonl",
    ):
        destination = output_root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_root / name, destination)
    (output_root / "COT_QUALITY_AUDIT.json").write_text(
        json.dumps(quality_report, indent=2) + "\n"
    )
    input_vla_hash = sha256_file(input_vla)
    output_vla_hash = sha256_file(output_vla)
    if input_vla_hash != output_vla_hash:
        raise ValueError("VLA output is not byte-identical to the source")
    expected_styles = {
        spec["name"]: rows // len(STYLE_SPECS)
        + (1 if index < rows % len(STYLE_SPECS) else 0)
        for index, spec in enumerate(STYLE_SPECS)
    }
    if dict(style_counts) != expected_styles:
        raise ValueError(
            f"style allocation mismatch: {dict(style_counts)} != {expected_styles}"
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "release_eligible": False,
        "release_gate": "experimental text-only rewrite; not re-reviewed by native visual dual critics",
        "created_at_utc": utc_now(),
        "input_root": str(input_root.resolve()),
        "output_root": str(output_root.resolve()),
        "model": DEFAULT_MODEL,
        "generation_resources": {
            "endpoints": sorted(generation_endpoints),
            "endpoint_count": len(generation_endpoints),
            "topology": "8 TP8 endpoints",
            "configured_gpus": len(generation_endpoints) * 8,
            "hidden_reasoning_generations": hidden_reasoning_generations,
        },
        "policy": {
            "style_assignment": "strict_source_row_index_mod_5",
            "styles": [dict(spec) for spec in STYLE_SPECS],
            "source_system_prompt_unchanged": True,
            "source_user_prompt_unchanged": True,
            "images_unchanged": True,
            "target_unchanged": True,
            "trajectory_metadata_unchanged": True,
            "only_assistant_think_regenerated": True,
            "source_reasoning_was_multimodally_audited": True,
            "rewrite_model_reinterpreted_images": False,
            "source_cot_had_dual_model_critics": True,
            "rewrite_validation": (
                "deterministic length, leakage, action/arm semantics, style, and "
                "source-similarity gates"
            ),
        },
        "counts": {
            "vlm_rows": rows,
            "style_rows": dict(style_counts),
            "accepted_rounds": {
                str(key): value for key, value in sorted(accepted_rounds.items())
            },
            "unique_reasoning_sha256": len(reasoning_hashes),
            "vla_episodes": len(json.loads(input_vla.read_text(encoding="utf-8"))),
        },
        "quality": {
            "word_count": {
                "min": min(word_counts),
                "mean": round(statistics.mean(word_counts), 3),
                "median": statistics.median(word_counts),
                "max": max(word_counts),
            },
            "lexical_similarity_to_source": {
                "mean": round(statistics.mean(similarities), 6),
                "max": max(similarities),
            },
            "fourgram_overlap_to_source": {
                "mean": round(statistics.mean(overlaps), 6),
                "max": max(overlaps),
            },
            "all_candidates_passed_rewrite_validation": True,
            "all_non_reasoning_vlm_fields_unchanged": True,
            "vla_byte_identical": True,
        },
        "source_hashes": {
            "data/vlm/train.jsonl": sha256_file(input_vlm),
            "data/vla/train.json": input_vla_hash,
        },
        "output_hashes": {
            "data/vlm/train.jsonl": sha256_file(output_vlm),
            "data/vla/train.json": output_vla_hash,
            "work/cot_style_annotations.jsonl": sha256_file(annotation_path),
        },
        "outputs": {
            "vlm_train": str(output_vlm),
            "vla_train": str(output_vla),
            "annotations": str(annotation_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(
        "# RoboTwin recovery CoT: five reasoning styles\n\n"
        "Experimental text-only rewrite, not a formal native release.\n\n"
        "The VLM system/user prompts, images, targets, and trajectory metadata "
        "are unchanged from the audited source release. Qwen3.8-Flash-Next "
        "regenerated only the assistant `<think>` reasoning using five balanced "
        "styles. The source reasoning had already passed multimodal dual-critic "
        "review; rewritten reasoning passed deterministic action/arm, leakage, "
        "style, length, and source-similarity gates. The VLA file is byte-identical "
        "to the source.\n",
        encoding="utf-8",
    )
    return report
