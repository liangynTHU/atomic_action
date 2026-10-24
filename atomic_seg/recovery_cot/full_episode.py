"""Full-episode CoT rows with dataset phases and model outcome judgments.

Unlike ``recovery_generation``, this pilot format covers the complete source
rollout, failure chain, and recovery trajectory.  Source-policy actions are
diagnostic context only; only audited recovery actions receive action targets.

The trajectory phase is authoritative supervision from the episode Parquet.
The vision-language judge decides whether the preceding atomic subtask
completed as intended.  ``current_status`` is written from the dataset phase,
not predicted by the judge.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

from atomic_seg.common.cot_format import word_count as _word_count
from atomic_seg.common.numeric import round_vector as _round_vector
from atomic_seg.io import load_json, save_json, write_jsonl
from atomic_seg.media import materialize_camera_frames
from atomic_seg.recovery_cot.io import recovery_state_layout, recovery_video_path
from atomic_seg.recovery_cot.quality import adapter_final_sentence

FULL_EPISODE_COT_SCHEMA = "full_episode_recovery_cot_row"
FULL_EPISODE_MANIFEST_SCHEMA = "full_episode_recovery_cot_manifest"
OUTCOMES = {
    "success",
    "partial",
    "failed_no_effect",
    "failed_different",
    "unknown",
}
STATUSES = {
    "rollout",
    "pre_failure",
    "attempt",
    "failure",
    "post_failure",
    "failure_observation",
    "failure_outcome",
    "recovery",
    "recovery_outcome",
}
PHASES = {"rollout", "failure", "recovery"}
RAW_PHASE_TO_PHASE = {
    "rollout": "rollout",
    "pre_failure": "rollout",
    "attempt": "failure",
    "failure_observation": "failure",
    "failure_outcome": "failure",
    "failure": "failure",
    "post_failure": "failure",
    "recovery": "recovery",
    "recovery_outcome": "recovery",
}
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
CHUNK_SIZE = 8


def _named_state(
    value: Sequence[Any],
    layout: Sequence[str],
) -> dict[str, float]:
    values = _round_vector(value)
    if len(values) != len(layout):
        raise ValueError("state and layout dimensions differ")
    return dict(zip(layout, values))


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "episode"


def _validate_judgment(value: Mapping[str, Any], context: str) -> None:
    outcome = str(value.get("previous_subtask_outcome", ""))
    status = str(value.get("current_status", ""))
    if outcome not in OUTCOMES:
        raise ValueError(f"{context}: unsupported outcome {outcome!r}")
    if status not in STATUSES:
        raise ValueError(f"{context}: unsupported current_status {status!r}")
    confidence = float(value.get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"{context}: confidence must be in [0,1]")
    for field in ("observed_result", "failure_analysis"):
        if not str(value.get(field, "")).strip():
            raise ValueError(f"{context}: {field} must be non-empty")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{context}: evidence must be non-empty")


def _opening(outcome: str) -> str:
    if outcome == "success":
        return "The previous atomic subtask was completed successfully."
    if outcome == "partial":
        return "The previous atomic subtask was only partially completed."
    if outcome == "failed_no_effect":
        return (
            "The previous atomic subtask was not completed successfully. "
            "The intended effect did not occur."
        )
    if outcome == "failed_different":
        return (
            "The previous atomic subtask was not completed successfully. "
            "The execution produced a different result from the intended effect."
        )
    if outcome == "unknown":
        raise ValueError(
            "full-episode recovery CoT requires a decisive previous-subtask "
            "outcome; unknown is not allowed"
        )
    raise ValueError(f"unsupported outcome {outcome!r}")


def _normalize_sentence(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text and text[-1] not in ".!?:;":
        text += "."
    return text


def _sentences(value: str) -> list[str]:
    text = re.sub(r"\s+", " ", value).strip()
    if not text:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def _without_model_status(value: Any, current_phase: str) -> str:
    """Drop cached status claims while preserving visible-state reasoning."""

    blocked = (
        "current status",
        "status remains",
        "trajectory status",
        "status is 'rollout'",
        'status is "rollout"',
    )
    failure_conflicts = (
        "it is not a failure",
        "no definitive failure",
        "no failure has occurred",
        "no failure is established",
        "without a confirmed failure",
    )
    kept: list[str] = []
    for sentence in _sentences(str(value)):
        lower = sentence.lower()
        if any(token in lower for token in blocked):
            continue
        if current_phase == "failure" and any(
            token in lower for token in failure_conflicts
        ):
            continue
        kept.append(sentence)
    return " ".join(kept)


def _trim_sentences(value: str, maximum_words: int) -> str:
    selected: list[str] = []
    for sentence in _sentences(value):
        candidate = " ".join((*selected, sentence))
        if selected and _word_count(candidate) > maximum_words:
            break
        selected.append(sentence)
        if _word_count(candidate) >= maximum_words:
            break
    if selected:
        return " ".join(selected)
    words = value.split()
    return " ".join(words[:maximum_words]).rstrip(" ,;:") + "."


def _phase_analysis(
    *,
    current_phase: str,
    previous_subtask: str,
    judgment: Mapping[str, Any],
) -> str:
    outcome = str(judgment["previous_subtask_outcome"])
    model_analysis = _normalize_sentence(judgment["failure_analysis"])
    reference = judgment.get("reference_failure_context")
    reference_analysis = ""
    if isinstance(reference, Mapping):
        summary = _normalize_sentence(reference.get("failure_summary", ""))
        evidence = [
            _normalize_sentence(item).rstrip(".")
            for item in reference.get("failure_evidence", [])
            if str(item).strip()
        ]
        reference_analysis = summary
        if evidence:
            reference_analysis += " The audited trajectory confirms that "
            reference_analysis += evidence[0][0].lower() + evidence[0][1:]
            reference_analysis += "."
    if current_phase == "rollout":
        if outcome in {"failed_no_effect", "failed_different", "partial"}:
            return model_analysis
        return (
            "No recovery-specific intervention is required at this boundary; "
            "the next atomic action can continue from the observed state."
        )
    if current_phase == "failure":
        if reference_analysis:
            return reference_analysis
        if outcome in {"failed_no_effect", "failed_different", "partial"}:
            return model_analysis
        if outcome == "success":
            return (
                "This local atomic motion succeeded, but it belongs to an "
                "attempt whose required task-level effect was not achieved or "
                "verified. The successful local motion therefore does not "
                "remove the need to diagnose and recover from the failed "
                "attempt."
            )
        return (
            "The episode is in the failure stage because the completed attempt "
            "did not establish the required task effect. The visible result is "
            "still insufficient to localize the exact physical cause, so the "
            "failure may involve alignment, contact, execution magnitude, "
            "premature release, or an effect that is not directly observable."
        )
    if current_phase == "recovery":
        if outcome in {"failed_no_effect", "failed_different"}:
            return (
                "This corrective step did not restore the condition required by "
                f"the original task. {model_analysis} The recovery plan must be "
                "revised from the newly observed object and gripper state "
                "instead of assuming the previous correction worked."
            )
        if outcome == "partial":
            return (
                "The correction made useful progress but did not yet restore "
                "all preconditions for the original task. The next step should "
                "preserve the recovered progress while addressing the remaining "
                "misalignment or incomplete interaction."
            )
        if outcome == "success":
            return (
                "This corrective atomic step restored an expected precondition "
                "or advanced the retry, but the earlier failure remains relevant "
                "until the original task effect is completed and verified."
            )
        return (
            "The episode is already following a recovery branch, but the effect "
            f"of the preceding correction, {previous_subtask.rstrip('.')}, is "
            "not yet visually conclusive. The next decision should avoid relying "
            "on an unverified correction and preserve a path to recheck the "
            "failed task effect."
        )
    raise ValueError(f"unsupported current_phase {current_phase!r}")


def _reference_failure_text(judgment: Mapping[str, Any]) -> str:
    reference = judgment.get("reference_failure_context")
    if not isinstance(reference, Mapping):
        return ""
    return _normalize_sentence(reference.get("failure_summary", ""))


def _fused_evidence(judgment: Mapping[str, Any]) -> str:
    evidence = [
        _normalize_sentence(item).rstrip(".")
        for item in judgment.get("evidence", [])
        if str(item).strip()
    ]
    if not evidence:
        return ""
    lowered = [item[0].lower() + item[1:] for item in evidence]
    if len(lowered) == 1:
        return f"This reading is supported by the fact that {lowered[0]}."
    if len(lowered) == 2:
        return (
            f"This reading is consistent across the views: {lowered[0]}; "
            f"and {lowered[1]}."
        )
    return (
        "This reading is consistent across the views: "
        + "; ".join(lowered[:-1])
        + f"; and {lowered[-1]}."
    )


def _reasoning(
    *,
    task: str,
    previous_subtask: str,
    judgment: Mapping[str, Any],
    current_phase: str,
    raw_phase: str,
    next_action: Mapping[str, Any] | None,
) -> str:
    outcome = str(judgment["previous_subtask_outcome"])
    selected_body = (
        judgment.get("action_reasoning_body") or judgment.get("reasoning_body")
        if next_action is not None
        else judgment.get("reasoning_body")
    )
    body = _without_model_status(selected_body or "", current_phase)
    if current_phase == "failure" and judgment.get("reference_failure_context"):
        body = ""
    if not body:
        if current_phase == "failure" and judgment.get("reference_failure_context"):
            body = (
                f"The intended atomic subtask was to "
                f"{previous_subtask.rstrip('.').lower()}. The current observation "
                "must be interpreted together with the completed attempt rather "
                "than as an isolated pose."
            )
        else:
            body = (
                f"The intended atomic subtask was to "
                f"{previous_subtask.rstrip('.').lower()}. Comparing the before "
                f"and after observations, "
                f"{str(judgment['observed_result']).rstrip('.')}."
            )

    if current_phase == "rollout":
        body = _trim_sentences(body, 103)
        reasoning = " ".join((_opening(outcome), body))
        if _word_count(reasoning) < 80:
            reasoning += " " + _normalize_sentence(judgment["observed_result"])
        if _word_count(reasoning) < 80:
            reasoning += " " + _phase_analysis(
                current_phase=current_phase,
                previous_subtask=previous_subtask,
                judgment=judgment,
            )
        return _trim_sentences(reasoning, 120)

    body_budget = 50 if judgment.get("reference_failure_context") else 92
    body = _trim_sentences(body, body_budget)
    observed = _normalize_sentence(judgment["observed_result"])
    observed = "At this boundary, " + observed[0].lower() + observed[1:]
    phase_description = (
        "This segment is part of the failed attempt"
        if current_phase == "failure" and raw_phase == "attempt"
        else (
            "The episode is now in the failure stage"
            if current_phase == "failure"
            else "The episode is now in recovery"
        )
    )
    analysis = _phase_analysis(
        current_phase=current_phase,
        previous_subtask=previous_subtask,
        judgment=judgment,
    )
    parts = [
        _opening(outcome),
        body,
        observed,
        f"{phase_description}. {analysis}",
    ]
    if current_phase == "recovery":
        reference_text = _reference_failure_text(judgment)
        if reference_text:
            parts.insert(3, reference_text)
    if next_action is not None:
        parts.append(
            "The next atomic subtask should be to "
            f"{str(next_action['sub_task']).rstrip('.').lower()}. This action "
            "should be evaluated from the resulting object contact, alignment, "
            "and task effect before any later correction assumes it succeeded."
        )
    elif current_phase == "failure":
        parts.append(
            "Before continuing, the controller should preserve the observed "
            "failure state and choose a correction that directly restores the "
            "missing task precondition."
        )
    else:
        parts.append(
            f"The original objective remains to {task.rstrip('.').lower()}, so "
            "the recovered state must still be checked against that complete "
            "goal rather than only against the latest local motion."
        )
    reasoning = " ".join(parts)
    if _word_count(reasoning) < 150:
        fused = _fused_evidence(judgment)
        if fused:
            parts.insert(3, fused)
            reasoning = " ".join(parts)
    if _word_count(reasoning) < 150:
        if current_phase == "failure":
            parts.append(
                "The correction should be selected from the actual mismatch: "
                "first re-establish a safe pose and reliable target relation, "
                "then retry the missing interaction with enough clearance, "
                "contact, or grasp security, and finally verify the intended "
                "object-level effect from a new observation before resuming the "
                "remaining task sequence safely."
            )
        else:
            parts.append(
                "Recovery progress should be assessed causally: confirm that "
                "the object and gripper reached the prerequisite state for the "
                "next motion, avoid repeating the condition that caused the "
                "failure, and keep verification separate from merely executing "
                "the commanded joint movement."
            )
        reasoning = " ".join(parts)
    if _word_count(reasoning) > 220:
        excess = _word_count(reasoning) - 215
        parts[1] = _trim_sentences(body, max(55, _word_count(body) - excess))
        reasoning = " ".join(parts)
    return reasoning


def _assistant_response(
    reasoning: str,
    judgment: Mapping[str, Any],
    current_status: str,
    raw_phase: str,
    failure_analysis: str,
    next_action: Mapping[str, Any] | None,
) -> str:
    answer = {
        "previous_subtask_outcome": judgment["previous_subtask_outcome"],
        "current_status": current_status,
        "current_phase": RAW_PHASE_TO_PHASE[raw_phase],
        "raw_phase": raw_phase,
        "failure_analysis": failure_analysis,
        "confidence": round(float(judgment["confidence"]), 3),
        "next_action": next_action,
    }
    return (
        f"<think>\n{reasoning}\n{adapter_final_sentence(next_action)}\n</think>\n\n"
        + json.dumps(
            answer,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _human_prompt(
    *,
    task: str,
    previous_subtask: str,
    before_frame: int,
    current_frame: int,
    before_state: Mapping[str, Any],
    current_state: Mapping[str, Any],
    state_delta: Sequence[float],
    image_roles: Sequence[str],
    action_requested: bool,
) -> str:
    request = (
        "Give a decisive success/partial/failure judgment for the previous "
        "subtask, analyze any failure, and then provide the next recovery action."
        if action_requested
        else (
            "Give a decisive success/partial/failure judgment for the previous "
            "subtask and analyze any failure. This is a diagnostic row; do not "
            "provide a next action."
        )
    )
    return "\n".join(
        [
            "".join("<image>" for _ in image_roles),
            f"Task: {task}",
            "Image roles: " + ", ".join(image_roles),
            f"Previous intended atomic subtask: {previous_subtask}",
            f"Previous decision frame: {before_frame}",
            "Previous robot state: "
            + json.dumps(before_state, ensure_ascii=False, separators=(",", ":")),
            f"Current decision frame: {current_frame}",
            "Current robot state: "
            + json.dumps(current_state, ensure_ascii=False, separators=(",", ":")),
            "Observed robot-state delta: "
            + json.dumps(
                _round_vector(state_delta),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            (
                "Infer previous_subtask_outcome from the visual/state evidence; "
                "do not assume it was successful merely because a new segment "
                "begins. Unknown is not allowed: choose success, partial, "
                "failed_no_effect, or failed_different. The current_status is "
                "supplied separately from the dataset phase column."
            ),
            request,
        ]
    )


def _load_episode_table(root: Path) -> dict[str, Any]:
    path = root / "data" / "chunk-000" / "episode_000000.parquet"
    columns = [
        "observation.state",
        "action",
        "timestamp",
        "phase",
        "action_is_supervised",
        "observation_only",
        "task_success",
    ]
    available = pq.read_schema(path).names
    table = pq.read_table(
        path,
        columns=[name for name in columns if name in available],
    )
    return {
        name: np.asarray(table.column(name).to_pylist()) for name in table.column_names
    }


def _phase_at(phases: Sequence[Any], frame: int) -> tuple[str, str]:
    if not 0 <= frame < len(phases):
        raise ValueError(f"phase frame {frame} is outside episode length")
    raw_phase = str(phases[frame])
    try:
        current_phase = RAW_PHASE_TO_PHASE[raw_phase]
    except KeyError as error:
        raise ValueError(f"unsupported dataset phase {raw_phase!r}") from error
    return raw_phase, current_phase


def _phase_sequence(
    phases: Sequence[Any],
    before_frame: int,
    current_frame: int,
) -> list[str]:
    output: list[str] = []
    for value in phases[before_frame : current_frame + 1]:
        label = str(value)
        if not output or output[-1] != label:
            output.append(label)
    return output


def _action_target(
    segment: Mapping[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
) -> dict[str, Any]:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    return {
        "guide_action": _round_vector(actions[end] - states[start]),
        "primary_action_verb": str(segment["primary_action_verb"]),
        "sub_task": str(segment["sub_task"]),
        "num_chunks": int(math.ceil((end - start + 1) / CHUNK_SIZE)),
        "recovery_role": str(segment["recovery_role"]),
    }


def _append_images(
    *,
    episode: Mapping[str, Any],
    assets: Mapping[tuple[str, int], Path],
    before_frame: int,
    current_frame: int,
    output_dir: Path,
) -> tuple[list[str], list[str], list[int]]:
    roles: list[str] = []
    paths: list[str] = []
    frames: list[int] = []
    temporal_frames = [("previous", before_frame)]
    if episode.get("include_midpoint_images") is True:
        temporal_frames.append(("intermediate", (before_frame + current_frame) // 2))
    temporal_frames.append(("current", current_frame))
    for prefix, frame in temporal_frames:
        for camera in CAMERAS:
            roles.append(f"{prefix}_{camera}")
            path = assets[(camera, frame)]
            paths.append(path.relative_to(output_dir).as_posix())
            frames.append(frame)
    return paths, roles, frames


def _row(
    *,
    episode: Mapping[str, Any],
    row_index: int,
    sample_type: str,
    previous_segment_id: str,
    previous_subtask: str,
    before_frame: int,
    current_frame: int,
    judgment: Mapping[str, Any],
    reference_failure_context: Mapping[str, Any] | None,
    next_action: Mapping[str, Any] | None,
    candidate_next_action: Mapping[str, Any] | None,
    action_supervision_reason: str,
    states: np.ndarray,
    phases: Sequence[Any],
    state_layout: Sequence[str],
    assets: Mapping[tuple[str, int], Path],
    output_dir: Path,
    phase_frame: int | None = None,
) -> dict[str, Any]:
    _validate_judgment(judgment, f"row {row_index}")
    before_state = _named_state(states[before_frame], state_layout)
    current_state = _named_state(states[current_frame], state_layout)
    images, image_roles, image_frames = _append_images(
        episode=episode,
        assets=assets,
        before_frame=before_frame,
        current_frame=current_frame,
        output_dir=output_dir,
    )
    label_frame = before_frame if phase_frame is None else int(phase_frame)
    raw_phase, current_phase = _phase_at(phases, label_frame)
    output_status = raw_phase
    raw_phase_sequence = _phase_sequence(phases, before_frame, current_frame)
    reasoning_judgment = dict(judgment)
    if reference_failure_context is not None:
        reasoning_judgment["reference_failure_context"] = dict(
            reference_failure_context
        )
    failure_analysis = _phase_analysis(
        current_phase=current_phase,
        previous_subtask=previous_subtask,
        judgment=reasoning_judgment,
    )
    reasoning = _reasoning(
        task=str(episode["task"]),
        previous_subtask=previous_subtask,
        judgment=reasoning_judgment,
        current_phase=current_phase,
        raw_phase=raw_phase,
        next_action=next_action,
    )
    row = {
        "schema": FULL_EPISODE_COT_SCHEMA,
        "release_eligible": False,
        "sample_id": (
            f"{_safe_component(str(episode['episode_id']))}__row_{row_index:03d}"
        ),
        "episode_id": episode["episode_id"],
        "episode_index": int(episode.get("episode_index", 0)),
        "row_index": row_index,
        "sample_type": sample_type,
        "task": episode["task"],
        "previous_segment_id": previous_segment_id,
        "previous_subtask": previous_subtask,
        "current_status": raw_phase,
        "current_phase": current_phase,
        "raw_phase": raw_phase,
        "raw_phase_sequence": raw_phase_sequence,
        "phase_source": "dataset_phase_column",
        "previous_subtask_outcome": judgment["previous_subtask_outcome"],
        "action_supervision": next_action is not None,
        "action_supervision_reason": action_supervision_reason,
        "loss_scope": (
            "judgment_phase_reasoning_and_recovery_action"
            if next_action is not None
            else "judgment_phase_and_failure_reasoning_only"
        ),
        "images": images,
        "image_roles": image_roles,
        "image_frames": image_frames,
        "previous_frame": before_frame,
        "current_frame": current_frame,
        "phase_frame": label_frame,
        "previous_state": before_state,
        "current_state": current_state,
        "observed_state_delta": _round_vector(
            states[current_frame] - states[before_frame]
        ),
        "model_judgment": dict(judgment),
        "reference_failure_context": (
            dict(reference_failure_context)
            if reference_failure_context is not None
            else None
        ),
        "next_action_target": next_action,
        "candidate_next_action": candidate_next_action,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You assess robot atomic-subtask outcomes from before/after "
                    "multiview observations and state. Infer whether the prior "
                    "subtask succeeded, partially completed, had no intended "
                    "effect, or produced a different effect. "
                    "The trajectory status is supplied from the dataset phase "
                    "column. Give a decisive previous-subtask outcome: success, "
                    "partial, failed_no_effect, or failed_different. Failure and "
                    "recovery reasoning should be detailed and physically "
                    "grounded. Never assume success from a segment boundary alone."
                    " Respond with one <think>...</think> block and then one JSON "
                    "object with previous_subtask_outcome, current_status, current_phase, "
                    "raw_phase, failure_analysis, confidence, and next_action. Set "
                    "next_action to null for a diagnostic-only sample."
                ),
            },
            {
                "role": "user",
                "content": _human_prompt(
                    task=str(episode["task"]),
                    previous_subtask=previous_subtask,
                    before_frame=before_frame,
                    current_frame=current_frame,
                    before_state=before_state,
                    current_state=current_state,
                    state_delta=states[current_frame] - states[before_frame],
                    image_roles=image_roles,
                    action_requested=next_action is not None,
                ),
            },
            {
                "role": "assistant",
                "content": _assistant_response(
                    reasoning,
                    judgment,
                    output_status,
                    raw_phase,
                    failure_analysis,
                    next_action,
                ),
            },
        ],
        "generation": {
            "judge_model": episode["judge_model"],
            "judge_mode": episode["judge_mode"],
            "judge_provenance": judgment.get("judge_provenance"),
            "judgment_generated_from_visible_evidence": True,
            "phase_source": "dataset_phase_column",
            "reference_failure_annotation_exposed_to_training_human": False,
            "failed_source_action_used_as_action_target": False,
        },
    }
    validate_full_episode_row(row)
    return row


def validate_full_episode_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != FULL_EPISODE_COT_SCHEMA:
        raise ValueError("unexpected full-episode recovery CoT schema")
    outcome = str(row.get("previous_subtask_outcome", ""))
    phase = str(row.get("current_phase", ""))
    raw_phase = str(row.get("raw_phase", ""))
    if outcome not in OUTCOMES - {"unknown"} or phase not in PHASES:
        raise ValueError("invalid model outcome or dataset phase label")
    if RAW_PHASE_TO_PHASE.get(raw_phase) != phase:
        raise ValueError("raw phase does not map to current phase")
    if row.get("phase_source") != "dataset_phase_column":
        raise ValueError("full-episode row requires dataset phase provenance")
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        raise ValueError("full-episode row requires system/user/assistant messages")
    human = str(messages[1].get("content", ""))
    assistant = str(messages[2].get("content", ""))
    images = row.get("images")
    if not isinstance(images, list) or len(images) not in (6, 9):
        raise ValueError(
            "full-episode row requires six endpoint images or nine "
            "start/middle/end images"
        )
    if human.count("<image>") != len(images):
        raise ValueError("full-episode image placeholders do not match")
    for leak in (
        f"current_phase: {phase}",
        f'"current_phase":"{phase}"',
        f'"previous_subtask_outcome":"{outcome}"',
    ):
        if leak in human:
            raise ValueError("model judgment label leaked into training human")
    match = re.fullmatch(
        r"\s*<think>\s*(.+?)\s*</think>\s*(\{.*\})\s*",
        assistant,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("assistant must contain think block and JSON")
    answer = json.loads(match.group(2))
    if answer["previous_subtask_outcome"] != outcome:
        raise ValueError("assistant outcome differs from model judgment")
    if answer["current_phase"] != phase:
        raise ValueError("assistant phase differs from dataset phase")
    expected_status = raw_phase
    if answer["current_status"] != expected_status:
        raise ValueError("assistant status differs from dataset phase")
    if answer["raw_phase"] != raw_phase:
        raise ValueError("assistant raw phase differs from dataset phase")
    if answer["next_action"] != row.get("next_action_target"):
        raise ValueError("assistant next action differs from target")
    if bool(answer["next_action"] is not None) != bool(row["action_supervision"]):
        raise ValueError("action supervision flag differs from assistant output")
    if row["action_supervision"] and not row.get("action_supervision_reason"):
        raise ValueError("action-supervised row requires an eligibility reason")
    reasoning = match.group(1)
    closing = adapter_final_sentence(row.get("next_action_target"))
    if not reasoning.rstrip().endswith(closing):
        raise ValueError("full-episode final action/diagnosis sentence mismatch")
    reasoning = reasoning.rstrip()[: -len(closing)].rstrip()
    if re.search(r"\bEvidence\s+\d+\s*:", reasoning, re.IGNORECASE):
        raise ValueError("numbered evidence templates are not allowed")
    word_count = _word_count(reasoning)
    minimum, maximum = (150, 230) if phase in {"failure", "recovery"} else (80, 125)
    if not minimum <= word_count <= maximum:
        raise ValueError(
            f"{phase} reasoning has {word_count} words, expected {minimum}-{maximum}"
        )


def _prepare_episode(
    entry: Mapping[str, Any],
    output_dir: Path,
    existing_asset_root: Path | None = None,
    reference_review_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(str(entry["data_root"])).resolve()
    source = _load_episode_table(root)
    states = np.asarray(source["observation.state"], dtype=np.float64)
    actions = np.asarray(source["action"], dtype=np.float64)
    phases = np.asarray(source["phase"], dtype=object)
    layout = recovery_state_layout(root)
    source_segments = entry.get("source_segments")
    recovery_segments = entry.get("recovery_segments")
    if not isinstance(source_segments, list) or not source_segments:
        raise ValueError("full episode requires source_segments")
    if not isinstance(recovery_segments, list) or not recovery_segments:
        raise ValueError("full episode requires recovery_segments")
    failure_judgment = entry.get("failure_judgment")
    terminal_judgment = entry.get("terminal_judgment")
    if not isinstance(failure_judgment, Mapping):
        raise ValueError("full episode requires failure_judgment")
    if not isinstance(terminal_judgment, Mapping):
        raise ValueError("full episode requires terminal_judgment")
    _validate_judgment(failure_judgment, "failure_judgment")
    _validate_judgment(terminal_judgment, "terminal_judgment")
    reference_failure: dict[str, Any] | None = None
    if reference_review_root is not None:
        matches: list[dict[str, Any]] = []
        episode_id = str(entry["episode_id"])
        episode_key = episode_id.removesuffix("_full_episode")
        candidates = list(
            reference_review_root.glob("episodes/*/metadata/episode_review.json")
        )
        for path in candidates:
            value = load_json(path)
            case = str(value.get("case", ""))
            if case == episode_key or case.endswith(f"_{episode_key}"):
                matches.append(value)
        if not matches:
            for path in candidates:
                value = load_json(path)
                if str(value.get("task")) == str(entry["task"]):
                    matches.append(value)
        if len(matches) == 1:
            review = matches[0]
            reference_failure = {
                "failure_type": review.get("failure_type"),
                "failure_summary": review.get("failure_summary"),
                "failure_evidence": review.get("failure_evidence", []),
                "source": "paired_recovery_episode_annotation",
            }

    requested_frames: set[int] = set()
    for segment in source_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        requested_frames.update((start, end))
        if entry.get("include_midpoint_images") is True:
            requested_frames.add((start + end) // 2)
    for segment in recovery_segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        requested_frames.update((start, end))
    if entry.get("include_midpoint_images") is True:
        for previous, current in zip(
            recovery_segments[:-1],
            recovery_segments[1:],
        ):
            requested_frames.add(
                (int(previous["start_frame"]) + int(current["start_frame"])) // 2
            )
    failure_frame = int(entry["failure_outcome_frame"])
    recovery_outcome_frame = int(entry["recovery_outcome_frame"])
    requested_frames.update((failure_frame, recovery_outcome_frame))
    if entry.get("include_midpoint_images") is True:
        requested_frames.add(
            (int(failure_judgment["before_frame"]) + failure_frame) // 2
        )
        requested_frames.add(
            (int(recovery_segments[-1]["start_frame"]) + recovery_outcome_frame) // 2
        )
    videos = {camera: recovery_video_path(root, 0, camera) for camera in CAMERAS}
    asset_root = output_dir / "assets" / _safe_component(str(entry["episode_id"]))
    assets: dict[tuple[str, int], Path] = {}
    reusable = (
        existing_asset_root / "assets" / _safe_component(str(entry["episode_id"]))
        if existing_asset_root is not None
        else None
    )
    if reusable is not None and reusable.is_dir():
        asset_root.mkdir(parents=True, exist_ok=True)
        missing: list[str] = []
        for camera in CAMERAS:
            for frame in requested_frames:
                name = f"{camera}_f{frame:06d}.jpg"
                source_asset = reusable / name
                if not source_asset.is_file():
                    missing.append(name)
                    continue
                target = asset_root / name
                if not target.exists():
                    shutil.copy2(source_asset, target)
                assets[(camera, frame)] = target
        if missing:
            raise FileNotFoundError(
                f"existing asset root is missing {len(missing)} requested "
                f"frames; first missing asset: {missing[0]}"
            )
    else:
        assets = materialize_camera_frames(
            videos,
            {camera: requested_frames for camera in CAMERAS},
            asset_root,
        )
    rows: list[dict[str, Any]] = []
    row_index = 0
    for segment in source_segments:
        judgment = segment.get("model_judgment")
        if not isinstance(judgment, Mapping):
            raise ValueError(
                f"source segment {segment.get('segment_id')} lacks judgment"
            )
        rows.append(
            _row(
                episode=entry,
                row_index=row_index,
                sample_type=(
                    "rollout_assessment"
                    if _phase_at(phases, int(segment["start_frame"]))[1] == "rollout"
                    else "failure_diagnosis"
                ),
                previous_segment_id=str(segment["segment_id"]),
                previous_subtask=str(segment["intended_subtask"]),
                before_frame=int(segment["start_frame"]),
                current_frame=int(segment["end_frame"]),
                judgment=judgment,
                reference_failure_context=(
                    reference_failure
                    if _phase_at(phases, int(segment["start_frame"]))[1] == "failure"
                    else None
                ),
                next_action=None,
                candidate_next_action=None,
                action_supervision_reason="source_action_is_diagnostic_only",
                states=states,
                phases=phases,
                state_layout=layout,
                assets=assets,
                output_dir=output_dir,
            )
        )
        row_index += 1

    first_recovery = recovery_segments[0]
    first_action = _action_target(first_recovery, states, actions)
    first_eligible = _phase_at(phases, failure_frame)[1] == "failure"
    rows.append(
        _row(
            episode=entry,
            row_index=row_index,
            sample_type=("recovery_action" if first_eligible else "recovery_diagnosis"),
            previous_segment_id=str(failure_judgment["previous_segment_id"]),
            previous_subtask=str(failure_judgment["previous_subtask"]),
            before_frame=int(failure_judgment["before_frame"]),
            current_frame=failure_frame,
            judgment=failure_judgment,
            reference_failure_context=reference_failure,
            next_action=first_action if first_eligible else None,
            candidate_next_action=first_action,
            action_supervision_reason=(
                "dataset_failure_phase_starts_reference_recovery"
                if first_eligible
                else "dataset_phase_does_not_start_recovery"
            ),
            states=states,
            phases=phases,
            state_layout=layout,
            assets=assets,
            output_dir=output_dir,
            phase_frame=failure_frame,
        )
    )
    row_index += 1
    for previous, current in zip(recovery_segments[:-1], recovery_segments[1:]):
        judgment = previous.get("model_judgment")
        if not isinstance(judgment, Mapping):
            raise ValueError(
                f"recovery segment {previous.get('segment_id')} lacks judgment"
            )
        candidate = _action_target(current, states, actions)
        eligible = _phase_at(phases, int(current["start_frame"]))[1] == "recovery"
        rows.append(
            _row(
                episode=entry,
                row_index=row_index,
                sample_type=("recovery_action" if eligible else "recovery_diagnosis"),
                previous_segment_id=str(previous["segment_id"]),
                previous_subtask=str(previous["sub_task"]),
                before_frame=int(previous["start_frame"]),
                current_frame=int(current["start_frame"]),
                judgment=judgment,
                reference_failure_context=reference_failure,
                next_action=candidate if eligible else None,
                candidate_next_action=candidate,
                action_supervision_reason=(
                    "dataset_recovery_phase_reference_action"
                    if eligible
                    else "dataset_phase_is_not_recovery"
                ),
                states=states,
                phases=phases,
                state_layout=layout,
                assets=assets,
                output_dir=output_dir,
            )
        )
        row_index += 1
    last_recovery = recovery_segments[-1]
    rows.append(
        _row(
            episode=entry,
            row_index=row_index,
            sample_type="terminal_recovery_assessment",
            previous_segment_id=str(last_recovery["segment_id"]),
            previous_subtask=str(last_recovery["sub_task"]),
            before_frame=int(last_recovery["start_frame"]),
            current_frame=recovery_outcome_frame,
            judgment=terminal_judgment,
            reference_failure_context=reference_failure,
            next_action=None,
            candidate_next_action=None,
            action_supervision_reason="terminal_assessment_has_no_next_action",
            states=states,
            phases=phases,
            state_layout=layout,
            assets=assets,
            output_dir=output_dir,
        )
    )
    summary = {
        "episode_id": entry["episode_id"],
        "task": entry["task"],
        "row_count": len(rows),
        "source_assessment_count": len(source_segments),
        "recovery_action_count": len(recovery_segments),
        "eligible_recovery_action_count": sum(
            bool(row["action_supervision"]) for row in rows
        ),
        "failure_row_count": sum(row["current_phase"] == "failure" for row in rows),
        "rollout_row_count": sum(row["current_phase"] == "rollout" for row in rows),
        "recovery_row_count": sum(row["current_phase"] == "recovery" for row in rows),
        "raw_phase_counts": {
            raw_phase: sum(row["raw_phase"] == raw_phase for row in rows)
            for raw_phase in sorted({str(row["raw_phase"]) for row in rows})
        },
        "phase_segmentation_note": (
            "Existing semantic atomic boundaries were retained. Any raw phase "
            "transition inside a row is preserved in raw_phase_sequence."
        ),
    }
    return rows, summary


def generate_full_episode_dataset(
    manifest_path: Path | str,
    output_dir: Path | str,
    existing_asset_root: Path | str | None = None,
    reference_review_root: Path | str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("schema") != FULL_EPISODE_MANIFEST_SCHEMA:
        raise ValueError("unexpected full-episode recovery manifest schema")
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("full-episode manifest requires episodes")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    reusable_assets = (
        Path(existing_asset_root) if existing_asset_root is not None else None
    )
    reference_reviews = (
        Path(reference_review_root) if reference_review_root is not None else None
    )
    for entry in entries:
        episode_rows, summary = _prepare_episode(
            entry,
            destination,
            reusable_assets,
            reference_reviews,
        )
        rows.extend(episode_rows)
        summaries.append(summary)
    write_jsonl(destination / "full_episode_recovery_cot.jsonl", rows)
    action_rows = [row for row in rows if row["action_supervision"]]
    diagnostic_rows = [row for row in rows if not row["action_supervision"]]
    write_jsonl(destination / "recovery_action_cot.jsonl", action_rows)
    write_jsonl(destination / "recovery_diagnosis_cot.jsonl", diagnostic_rows)
    report = {
        "schema": "full_episode_recovery_cot_generation_manifest",
        "release_eligible": False,
        "release_gate": "experimental outcome/phase diagnostics, not a native planner release",
        "source_manifest": str(Path(manifest_path).resolve()),
        "episode_count": len(entries),
        "row_count": len(rows),
        "judge_models": sorted({str(entry["judge_model"]) for entry in entries}),
        "judge_modes": sorted({str(entry["judge_mode"]) for entry in entries}),
        "phase_counts": {
            phase: sum(row["current_phase"] == phase for row in rows)
            for phase in sorted(PHASES)
        },
        "raw_phase_counts": {
            raw_phase: sum(row["raw_phase"] == raw_phase for row in rows)
            for raw_phase in sorted({str(row["raw_phase"]) for row in rows})
        },
        "status_counts": {
            status: sum(row["current_status"] == status for row in rows)
            for status in sorted({str(row["current_status"]) for row in rows})
        },
        "outcome_counts": {
            outcome: sum(row["previous_subtask_outcome"] == outcome for row in rows)
            for outcome in sorted(OUTCOMES)
        },
        "action_supervision_rows": len(action_rows),
        "diagnostic_only_rows": len(diagnostic_rows),
        "outputs": {
            "all_rows": "full_episode_recovery_cot.jsonl",
            "action_rows": "recovery_action_cot.jsonl",
            "diagnostic_rows": "recovery_diagnosis_cot.jsonl",
        },
        "failed_source_actions_used_as_targets": False,
        "phase_source": "dataset_phase_column",
        "action_supervision_policy": (
            "reference recovery actions are supervised only at dataset-labeled "
            "failure/recovery decision origins; model outcome judgments remain "
            "independent supervision and never define the trajectory phase"
        ),
        "episodes": summaries,
    }
    save_json(destination / "manifest.json", report)
    return rows, report
