#!/usr/bin/env python3
"""Blind-judge full recovery episodes with a local multimodal Qwen service."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from atomic_seg.common.model_client import multimodal_content, post_chat
from atomic_seg.common.numeric import round_vector as _round_vector
from atomic_seg.io import load_json, save_json
from atomic_seg.recovery_cot.full_episode import RAW_PHASE_TO_PHASE, STATUSES
from atomic_seg.recovery_cot.io import recovery_state_layout

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DECISIVE_OUTCOMES = {
    "success",
    "partial",
    "failed_no_effect",
    "failed_different",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument(
        "--reference-review-root",
        type=Path,
        default=None,
        help="Optional complete-episode review bundle with audited failures.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8007/v1/chat/completions",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--episode-id",
        action="append",
        default=[],
        help="Only judge selected episode_id values; may be repeated.",
    )
    parser.add_argument(
        "--skip-action-reasoning",
        action="store_true",
        help=(
            "Use the blind judgment reasoning plus deterministic action "
            "rendering instead of a target-conditioned second model call."
        ),
    )
    return parser.parse_args()


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "episode"


def _named_state(
    value: Sequence[Any],
    layout: Sequence[str],
) -> dict[str, float]:
    return dict(zip(layout, _round_vector(value)))


def _load_episode_data(
    root: Path,
) -> tuple[np.ndarray, list[str], list[bool]]:
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    table = pq.read_table(
        path,
        columns=["observation.state", "phase", "task_success"],
    )
    return (
        np.asarray(table.column("observation.state").to_pylist(), dtype=float),
        [str(value) for value in table.column("phase").to_pylist()],
        [bool(value) for value in table.column("task_success").to_pylist()],
    )


def _dataset_status(phases: Sequence[str], frame: int) -> tuple[str, str]:
    raw_phase = str(phases[frame])
    try:
        return raw_phase, RAW_PHASE_TO_PHASE[raw_phase]
    except KeyError as error:
        raise ValueError(f"unsupported dataset phase {raw_phase!r}") from error


def _reference_failure(
    review_root: Path | None,
    episode: Mapping[str, Any],
) -> dict[str, Any] | None:
    if review_root is None:
        return None
    episode_key = str(episode["episode_id"]).removesuffix("_full_episode")
    candidates = list(review_root.glob("episodes/*/metadata/episode_review.json"))
    matches = []
    for path in candidates:
        value = load_json(path)
        case = str(value.get("case", ""))
        if case == episode_key or case.endswith(f"_{episode_key}"):
            matches.append(value)
    if not matches:
        for path in candidates:
            value = load_json(path)
            if str(value.get("task")) == str(episode["task"]):
                matches.append(value)
    if len(matches) != 1:
        raise ValueError(
            f"{episode['episode_id']}: expected one reference review, "
            f"found {len(matches)}"
        )
    review = matches[0]
    return {
        "failure_type": review.get("failure_type"),
        "failure_summary": review.get("failure_summary"),
        "failure_evidence": review.get("failure_evidence", []),
    }


def _images(
    asset_root: Path,
    episode_id: str,
    before_frame: int,
    current_frame: int,
    include_midpoint: bool,
) -> tuple[list[Path], list[str]]:
    root = asset_root / _safe_component(episode_id)
    paths: list[Path] = []
    roles: list[str] = []
    temporal_frames = [("previous", before_frame)]
    if include_midpoint:
        temporal_frames.append(("intermediate", (before_frame + current_frame) // 2))
    temporal_frames.append(("current", current_frame))
    for prefix, frame in temporal_frames:
        for camera in CAMERAS:
            path = root / f"{camera}_f{frame:06d}.jpg"
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
            roles.append(f"{prefix}_{camera}")
    return paths, roles


def _extract_json(value: str) -> dict[str, Any]:
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match is None:
            raise ValueError("model response does not contain JSON")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("model response JSON must be an object")
    return parsed


def _validate_judgment(
    value: Mapping[str, Any],
    *,
    current_status: str,
) -> dict[str, Any]:
    result = {
        "previous_subtask_outcome": str(value.get("previous_subtask_outcome", "")),
        "current_status": current_status,
        "observed_result": str(value.get("observed_result", "")).strip(),
        "failure_analysis": str(value.get("failure_analysis", "")).strip(),
        "evidence": [
            str(item).strip() for item in value.get("evidence", []) if str(item).strip()
        ],
        "confidence": float(value.get("confidence", -1.0)),
        "reasoning_body": str(value.get("reasoning_body", "")).strip(),
    }
    if not result["reasoning_body"]:
        for alias in ("reasoning", "analysis", "explanation"):
            candidate = str(value.get(alias, "")).strip()
            if candidate:
                result["reasoning_body"] = candidate
                break
    if not result["failure_analysis"]:
        for alias in ("failure_reason", "status_analysis", "diagnosis"):
            candidate = str(value.get(alias, "")).strip()
            if candidate:
                result["failure_analysis"] = candidate
                break
    if not result["observed_result"]:
        for alias in ("observed_outcome", "observation_summary"):
            candidate = str(value.get(alias, "")).strip()
            if candidate:
                result["observed_result"] = candidate
                break
    if result["previous_subtask_outcome"] not in DECISIVE_OUTCOMES:
        raise ValueError("invalid previous_subtask_outcome")
    if result["current_status"] not in STATUSES:
        raise ValueError("invalid current_status")
    if not 0.0 <= result["confidence"] <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    normalized_fields: list[str] = []
    if not result["failure_analysis"]:
        status = result["current_status"]
        if status == "rollout":
            result["failure_analysis"] = (
                "No failure condition is evident at this decision boundary."
            )
        elif status == "recovery":
            result["failure_analysis"] = (
                "The episode remains on a corrective branch; the next boundary "
                "must verify whether recovery progress is sufficient."
            )
        else:
            result["failure_analysis"] = (
                "The visible result does not satisfy the intended atomic "
                "subtask and requires a new corrective decision."
            )
        normalized_fields.append("failure_analysis")
    for field in ("observed_result", "reasoning_body"):
        if not result[field]:
            raise ValueError(f"missing {field}")
    if not result["evidence"]:
        raise ValueError("at least one evidence statement is required")
    word_count = len(re.findall(r"\b[\w'-]+\b", result["reasoning_body"]))
    minimum = 60
    if word_count < minimum:
        raise ValueError(
            f"reasoning_body has {word_count} words; expected at least {minimum}"
        )
    result["normalization"] = {
        "fields": normalized_fields,
        "outcome_or_status_changed": False,
    }
    return result


def _validate_action_reasoning(value: Mapping[str, Any]) -> str:
    body = str(value.get("reasoning_body", "")).strip()
    if not body:
        raise ValueError("missing action reasoning_body")
    word_count = len(re.findall(r"\b[\w'-]+\b", body))
    if word_count < 80:
        raise ValueError(
            f"action reasoning_body has {word_count} words; expected at least 80"
        )
    return body


def _post(
    *,
    endpoint: str,
    model: str,
    text: str,
    image_paths: Sequence[Path],
    image_roles: Sequence[str],
    timeout: float,
    retries: int,
    max_tokens: int,
    validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[Any, str]:
    content = multimodal_content(text, image_paths, image_roles)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a strict robot trajectory outcome judge. Use only "
                    "the supplied before/after images, robot states, intended "
                    "subtask, authoritative dataset phase, audited episode "
                    "failure context, and prior outcome history. You must make a "
                    "decisive best-supported success/partial/failure judgment; "
                    "never return unknown. Return JSON only."
                ),
            },
            {"role": "user", "content": content},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            body = post_chat(endpoint=endpoint, payload=payload, timeout=timeout)
            raw = body["choices"][0]["message"]["content"]
            if isinstance(raw, list):
                raw = "".join(
                    str(item.get("text", "")) for item in raw if isinstance(item, dict)
                )
            parsed = _extract_json(str(raw))
            validated = validator(parsed) if validator is not None else parsed
            return validated, str(raw)
        except (
            urllib.error.URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"Qwen request failed after {retries} attempts") from last_error


def _history_text(history: Sequence[Mapping[str, Any]]) -> str:
    visible = list(history[-6:])
    return json.dumps(visible, ensure_ascii=False, separators=(",", ":"))


def _judge_prompt(
    *,
    task: str,
    intended_subtask: str,
    before_frame: int,
    current_frame: int,
    before_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    state_delta: Sequence[float],
    history: Sequence[Mapping[str, Any]],
    current_status: str,
    raw_phase: str,
    reference_failure: Mapping[str, Any] | None,
    task_success_before: bool,
    task_success_current: bool,
) -> str:
    return "\n".join(
        [
            f"Task: {task}",
            f"Previous intended atomic subtask: {intended_subtask}",
            (
                "Robot-state layout: [left_waist, left_shoulder, left_elbow, "
                "left_forearm_roll, left_wrist_angle, left_wrist_rotate, "
                "left_gripper, right_waist, right_shoulder, right_elbow, "
                "right_forearm_roll, right_wrist_angle, right_wrist_rotate, "
                "right_gripper]. Gripper values are approximately 1=open and "
                "0=closed."
            ),
            f"Previous frame: {before_frame}",
            "Previous state: "
            + json.dumps(before_state, ensure_ascii=False, separators=(",", ":")),
            f"Current frame: {current_frame}",
            "Current state: "
            + json.dumps(current_state, ensure_ascii=False, separators=(",", ":")),
            "Observed state delta: "
            + json.dumps(
                _round_vector(state_delta),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Prior model-judgment history: " + _history_text(history),
            f"Authoritative dataset phase: {current_status}",
            f"Raw dataset phase: {raw_phase}",
            f"Overall task-success signal at previous frame: {task_success_before}",
            f"Overall task-success signal at current frame: {task_success_current}",
            (
                "Audited episode failure context: "
                + json.dumps(
                    reference_failure,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                if reference_failure is not None
                else "No audited failure context applies to this rollout row."
            ),
            (
                "The dataset phase is authoritative and must not be predicted or "
                "changed. Judge only whether the previous intended atomic "
                "subtask achieved its own intended result at this boundary."
            ),
            (
                "You must give a decisive best-supported label; unknown is not "
                "allowed. Use success when the intended local motion/effect is "
                "achieved, partial when meaningful progress is made but a stated "
                "requirement remains unmet, failed_no_effect when the intended "
                "effect is absent, and failed_different when execution produces "
                "a different physical result. When direct visual confirmation is "
                "weak, use the phase, temporal history, robot state, and audited "
                "episode context to choose the most likely label and lower the "
                "confidence instead of returning uncertainty. A local motion may "
                "succeed even inside a failed overall attempt. The overall "
                "task-success signal is decisive for subtasks that explicitly "
                "claim to complete or verify the full task, but a false signal "
                "does not by itself make an intermediate positioning motion fail."
            ),
            (
                "Return exactly one JSON object with keys: "
                "previous_subtask_outcome (success|partial|failed_no_effect|"
                "failed_different), observed_result, failure_analysis, "
                "evidence (2-4 concise strings), confidence (0..1), and "
                "reasoning_body. reasoning_body must be 70-120 words for rollout "
                "or 140-230 words for failure/recovery, grounded in visible "
                "before/after evidence. Do not output current_status and do not "
                "include a canned first sentence."
            ),
        ]
    )


def _action_prompt(
    *,
    task: str,
    intended_subtask: str,
    judgment: Mapping[str, Any],
    next_action: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    before_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    state_delta: Sequence[float],
) -> str:
    target = {
        key: next_action[key]
        for key in ("primary_action_verb", "sub_task", "recovery_role")
    }
    return "\n".join(
        [
            f"Task: {task}",
            f"Previous intended atomic subtask: {intended_subtask}",
            ("Robot-state gripper convention: approximately 1=open and 0=closed."),
            "Blind model judgment: "
            + json.dumps(judgment, ensure_ascii=False, separators=(",", ":")),
            "Prior judgment history: " + _history_text(history),
            "Previous state: "
            + json.dumps(before_state, ensure_ascii=False, separators=(",", ":")),
            "Current state: "
            + json.dumps(current_state, ensure_ascii=False, separators=(",", ":")),
            "Observed state delta: "
            + json.dumps(
                _round_vector(state_delta),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "Supervised next recovery action semantics: "
            + json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            (
                "Write a detailed 150-260 word reasoning_body that first "
                "explains the current physical state and any failure, then "
                "explains why the supervised next recovery action is appropriate. "
                "Do not alter the blind outcome/status judgment, do not claim the "
                "next action has already succeeded, and do not include a canned "
                'first sentence. Return JSON only: {"reasoning_body":"..."}.'
            ),
        ]
    )


def _cache_records(path: Path) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return output
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                output[str(value["cache_key"])] = value
    return output


def _append_cache(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"
        )


def _target(segment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: segment[key]
        for key in ("primary_action_verb", "sub_task", "recovery_role")
    }


def main() -> None:
    args = parse_args()
    source = load_json(args.manifest)
    if source.get("schema") != "full_episode_recovery_cot_manifest":
        raise ValueError("unexpected skeleton manifest schema")
    episodes = source.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("skeleton manifest requires episodes")
    if args.episode_id:
        wanted = set(args.episode_id)
        episodes = [
            episode for episode in episodes if str(episode.get("episode_id")) in wanted
        ]
        missing = wanted - {str(episode.get("episode_id")) for episode in episodes}
        if missing:
            raise ValueError(f"unknown episode ids: {sorted(missing)}")
    cache = _cache_records(args.cache)
    completed_rows = 0
    output_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        include_midpoint = episode.get("include_midpoint_images") is True
        cache_suffix = ":decisive_v5"
        if include_midpoint:
            cache_suffix += ":temporal3"
        states, phases, task_success = _load_episode_data(Path(episode["data_root"]))
        layout = recovery_state_layout(episode["data_root"])
        reference_failure = _reference_failure(
            args.reference_review_root,
            episode,
        )
        history: list[dict[str, Any]] = []
        output_episode = {
            key: value
            for key, value in episode.items()
            if key
            not in {
                "source_segments",
                "recovery_segments",
                "failure_judgment",
                "terminal_judgment",
            }
        }
        output_episode["judge_model"] = args.model
        output_episode["judge_mode"] = (
            "qwen_phase_conditioned_decisive_single_stage"
            if args.skip_action_reasoning
            else "qwen_phase_conditioned_decisive_two_stage"
        )
        output_sources: list[dict[str, Any]] = []

        def judge(
            key: str,
            intended: str,
            before: int,
            current: int,
            *,
            phase_frame: int | None = None,
        ) -> dict[str, Any]:
            nonlocal completed_rows
            label_frame = before if phase_frame is None else phase_frame
            raw_phase, current_status = _dataset_status(phases, label_frame)
            result: dict[str, Any] | None = None
            if key in cache:
                cached = dict(cache[key]["judgment"])
                try:
                    result = _validate_judgment(
                        cached,
                        current_status=current_status,
                    )
                except ValueError:
                    result = None
            if result is None:
                paths, roles = _images(
                    args.asset_root,
                    episode_id,
                    before,
                    current,
                    include_midpoint,
                )

                def validate_with_context(value: Mapping[str, Any]) -> dict[str, Any]:
                    return _validate_judgment(
                        value,
                        current_status=current_status,
                    )

                result, raw = _post(
                    endpoint=args.endpoint,
                    model=args.model,
                    text=_judge_prompt(
                        task=str(episode["task"]),
                        intended_subtask=intended,
                        before_frame=before,
                        current_frame=current,
                        before_state=_named_state(states[before], layout),
                        current_state=_named_state(states[current], layout),
                        state_delta=states[current] - states[before],
                        history=history,
                        current_status=current_status,
                        raw_phase=raw_phase,
                        reference_failure=(
                            reference_failure
                            if current_status in {"failure", "recovery"}
                            else None
                        ),
                        task_success_before=task_success[before],
                        task_success_current=task_success[current],
                    ),
                    image_paths=paths,
                    image_roles=roles,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_tokens=1000,
                    validator=validate_with_context,
                )
                result["judge_provenance"] = {
                    "model": args.model,
                    "mode": "phase_conditioned_decisive_multimodal_outcome",
                    "raw_response": raw,
                }
                record = {"cache_key": key, "judgment": result}
                _append_cache(args.cache, record)
                cache[key] = record
            completed_rows += 1
            print(
                f"judged={completed_rows} key={key} "
                f"outcome={result['previous_subtask_outcome']} "
                f"status={result['current_status']}",
                flush=True,
            )
            history.append(
                {
                    "subtask": intended,
                    "outcome": result["previous_subtask_outcome"],
                    "status": result["current_status"],
                    "observed_result": result["observed_result"],
                }
            )
            if args.max_rows is not None and completed_rows >= args.max_rows:
                raise StopIteration
            return result

        try:
            for segment in episode["source_segments"]:
                clean = {
                    key: value
                    for key, value in segment.items()
                    if key != "model_judgment"
                }
                cache_key = f"{episode_id}:source:{segment['segment_id']}{cache_suffix}"
                clean["model_judgment"] = judge(
                    cache_key,
                    str(segment["intended_subtask"]),
                    int(segment["start_frame"]),
                    int(segment["end_frame"]),
                )
                output_sources.append(clean)
            output_episode["source_segments"] = output_sources

            failure_structure = episode["failure_judgment"]
            failure_key = f"{episode_id}:failure_boundary{cache_suffix}"
            failure_result = judge(
                failure_key,
                str(failure_structure["previous_subtask"]),
                int(failure_structure["before_frame"]),
                int(episode["failure_outcome_frame"]),
                phase_frame=int(episode["failure_outcome_frame"]),
            )
            failure_result.update(
                {
                    "previous_segment_id": failure_structure["previous_segment_id"],
                    "previous_subtask": failure_structure["previous_subtask"],
                    "before_frame": failure_structure["before_frame"],
                }
            )
            raw_recovery = episode["recovery_segments"]
            first_key = f"{episode_id}:failure_boundary:action_reasoning{cache_suffix}"
            if args.skip_action_reasoning:
                first_body = ""
            elif first_key in cache:
                first_body = str(cache[first_key]["action_reasoning_body"])
            else:
                paths, roles = _images(
                    args.asset_root,
                    episode_id,
                    int(failure_structure["before_frame"]),
                    int(episode["failure_outcome_frame"]),
                    include_midpoint,
                )
                first_body, raw = _post(
                    endpoint=args.endpoint,
                    model=args.model,
                    text=_action_prompt(
                        task=str(episode["task"]),
                        intended_subtask=str(failure_structure["previous_subtask"]),
                        judgment=failure_result,
                        next_action=_target(raw_recovery[0]),
                        history=history,
                        before_state=_named_state(
                            states[int(failure_structure["before_frame"])],
                            layout,
                        ),
                        current_state=_named_state(
                            states[int(episode["failure_outcome_frame"])],
                            layout,
                        ),
                        state_delta=(
                            states[int(episode["failure_outcome_frame"])]
                            - states[int(failure_structure["before_frame"])]
                        ),
                    ),
                    image_paths=paths,
                    image_roles=roles,
                    timeout=args.timeout,
                    retries=args.retries,
                    max_tokens=1000,
                    validator=_validate_action_reasoning,
                )
                first_record = {
                    "cache_key": first_key,
                    "action_reasoning_body": first_body,
                    "raw_response": raw,
                }
                _append_cache(args.cache, first_record)
                cache[first_key] = first_record
            if first_body:
                failure_result["action_reasoning_body"] = first_body
            output_episode["failure_judgment"] = failure_result

            output_recovery: list[dict[str, Any]] = []
            for index, segment in enumerate(raw_recovery):
                clean = {
                    key: value
                    for key, value in segment.items()
                    if key != "model_judgment"
                }
                if index < len(raw_recovery) - 1:
                    current = raw_recovery[index + 1]
                    cache_key = (
                        f"{episode_id}:recovery:{segment['segment_id']}{cache_suffix}"
                    )
                    result = judge(
                        cache_key,
                        str(segment["sub_task"]),
                        int(segment["start_frame"]),
                        int(current["start_frame"]),
                    )
                    action_key = f"{cache_key}:action_reasoning"
                    if args.skip_action_reasoning:
                        action_body = ""
                    elif action_key in cache:
                        action_body = str(cache[action_key]["action_reasoning_body"])
                    else:
                        paths, roles = _images(
                            args.asset_root,
                            episode_id,
                            int(segment["start_frame"]),
                            int(current["start_frame"]),
                            include_midpoint,
                        )
                        action_body, raw = _post(
                            endpoint=args.endpoint,
                            model=args.model,
                            text=_action_prompt(
                                task=str(episode["task"]),
                                intended_subtask=str(segment["sub_task"]),
                                judgment=result,
                                next_action=_target(current),
                                history=history,
                                before_state=_named_state(
                                    states[int(segment["start_frame"])],
                                    layout,
                                ),
                                current_state=_named_state(
                                    states[int(current["start_frame"])],
                                    layout,
                                ),
                                state_delta=(
                                    states[int(current["start_frame"])]
                                    - states[int(segment["start_frame"])]
                                ),
                            ),
                            image_paths=paths,
                            image_roles=roles,
                            timeout=args.timeout,
                            retries=args.retries,
                            max_tokens=1000,
                            validator=_validate_action_reasoning,
                        )
                        action_record = {
                            "cache_key": action_key,
                            "action_reasoning_body": action_body,
                            "raw_response": raw,
                        }
                        _append_cache(args.cache, action_record)
                        cache[action_key] = action_record
                    if action_body:
                        result["action_reasoning_body"] = action_body
                    clean["model_judgment"] = result
                output_recovery.append(clean)
            output_episode["recovery_segments"] = output_recovery

            last = raw_recovery[-1]
            terminal_key = f"{episode_id}:terminal{cache_suffix}"
            terminal_result = judge(
                terminal_key,
                str(last["sub_task"]),
                int(last["start_frame"]),
                int(episode["recovery_outcome_frame"]),
            )
            output_episode["terminal_judgment"] = terminal_result
            output_episodes.append(output_episode)
        except StopIteration:
            break
    output = {
        "schema": "full_episode_recovery_cot_manifest",
        "judgment_source": {
            "model": args.model,
            "endpoint": args.endpoint,
            "mode": (
                "phase_conditioned_decisive_single_stage"
                if args.skip_action_reasoning
                else "phase_conditioned_decisive_two_stage"
            ),
            "dataset_phase_used_in_prompt": True,
            "audited_failure_context_used_in_prompt": (
                args.reference_review_root is not None
            ),
        },
        "episodes": output_episodes,
    }
    save_json(args.output_manifest, output)
    print(f"wrote={args.output_manifest} episodes={len(output_episodes)}")


if __name__ == "__main__":
    main()
