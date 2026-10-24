"""Segment-level CoT generation for audited recovery trajectories.

The success-only generator deliberately does not call this module.  Recovery
rows have a separate schema because they must expose and classify the previous
atomic subtask outcome before choosing the next corrective action.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from atomic_seg.common.model_client import (
    _response_parts,
    multimodal_content,
    post_chat,
)
from atomic_seg.common.numeric import round_vector as _round_vector
from atomic_seg.media import materialize_camera_frames
from atomic_seg.recovery_cot.io import (
    RecoveryEpisode,
    RecoverySegment,
    load_recovery_source,
    recovery_video_path,
    validate_recovery_segment_rows,
)
from atomic_seg.recovery_cot.quality import adapter_final_sentence
from atomic_seg.cot_data.styles import STYLE_SPECS

RECOVERY_ACTION_DATASET_SCHEMA = "atomic_recovery_action_dataset"
RECOVERY_ACTION_EPISODE_SCHEMA = "atomic_recovery_action_episode"
RECOVERY_COT_DATASET_SCHEMA = "atomic_recovery_cot_training_row"
RECOVERY_CHUNK_SIZE = 8

SUCCESS_PREFIX = "The previous atomic subtask was completed successfully."
FAILURE_PREFIX = "The previous atomic subtask was not completed successfully."
PARTIAL_PREFIX = "The previous atomic subtask was only partially completed."
UNKNOWN_PREFIX = (
    "Whether the previous atomic subtask was completed successfully is uncertain."
)


def _named_state(
    value: Sequence[Any],
    layout: Sequence[str],
) -> dict[str, float]:
    values = _round_vector(value)
    if len(values) != len(layout):
        raise ValueError(f"state has {len(values)} values but layout has {len(layout)}")
    return dict(zip(layout, values))


def _previous_outcome_prefix(status: str) -> str:
    if status == "success":
        return SUCCESS_PREFIX
    if status in {"failed_no_effect", "failed_different"}:
        return FAILURE_PREFIX
    if status == "partial":
        return PARTIAL_PREFIX
    if status == "unknown":
        return UNKNOWN_PREFIX
    raise ValueError(f"unsupported previous outcome status {status!r}")


def _previous_outcome_detail(status: str) -> str:
    if status == "failed_no_effect":
        return "The intended effect did not occur."
    if status == "failed_different":
        return "The execution produced a different result from the intended effect."
    return ""


def _strip_outcome_prefix(reasoning: str) -> str:
    value = reasoning.strip()
    for prefix in (
        SUCCESS_PREFIX,
        FAILURE_PREFIX,
        PARTIAL_PREFIX,
        UNKNOWN_PREFIX,
    ):
        if value.startswith(prefix):
            value = value[len(prefix) :].lstrip()
    return value


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return result or "recovery_episode"


@dataclass(frozen=True)
class RecoveryReasoningRequest:
    sample_id: str
    task: str
    recovery_level: int
    recovery_level_name: str
    failure_type: str
    previous_outcome: Mapping[str, Any]
    current_state: Mapping[str, Any]
    history_state: Mapping[str, Any] | None
    outcome_state: Mapping[str, Any] | None
    recovery_history: Sequence[Mapping[str, Any]]
    target: Mapping[str, Any]
    image_paths: Sequence[Path]
    image_roles: Sequence[str]
    style_id: int = 0


class RecoveryReasoner(Protocol):
    name: str

    def generate(self, request: RecoveryReasoningRequest) -> str:
        """Return recovery reasoning after the fixed outcome sentence."""


class RecoveryTemplateReasoner:
    """Deterministic, annotation-grounded smoke-test reasoner."""

    name = "deterministic_recovery_annotation_template"

    def generate(self, request: RecoveryReasoningRequest) -> str:
        previous = request.previous_outcome
        status = str(previous["status"])
        observed = str(previous.get("observed_outcome", "")).strip()
        mismatch = str(previous.get("mismatch", "")).strip()
        if status == "success":
            assessment = (
                observed
                or "The prior recovery motion reached its audited decision boundary."
            )
        elif status == "failed_no_effect":
            assessment = observed or "The intended state change was not observed."
        elif status == "failed_different":
            assessment = observed or "The observed result differs from the intent."
        elif status == "partial":
            assessment = observed or "Only part of the intended effect is visible."
        elif status == "unknown":
            assessment = (
                observed
                or "The available evidence does not establish the prior result."
            )
        else:
            raise ValueError(f"unsupported previous outcome status {status!r}")
        mismatch_text = f" {mismatch}" if mismatch else ""
        return (
            f"{assessment}{mismatch_text} The recovery remains at "
            f"{request.recovery_level_name} (L{request.recovery_level}) for "
            f"failure type {request.failure_type}. The next corrective motion "
            f"is: {request.target['sub_task']} This describes the supervised "
            "motion without claiming a later effect before it is observed."
        )


class OpenAICompatibleRecoveryReasoner:
    """Recovery reasoner backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"openai_compatible_recovery:{model}"

    def generate(self, request: RecoveryReasoningRequest) -> str:
        target_semantics = {
            key: request.target[key]
            for key in (
                "primary_action_verb",
                "sub_task",
                "recovery_role",
            )
        }
        text = "\n".join(
            [
                f"Task: {request.task}",
                f"Reasoning organization: {STYLE_SPECS[request.style_id]['brief']}",
                (
                    "Recovery classification: "
                    f"L{request.recovery_level} "
                    f"{request.recovery_level_name}"
                ),
                f"Failure type: {request.failure_type}",
                "Previous outcome: "
                + json.dumps(
                    request.previous_outcome,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Recovery history: "
                + json.dumps(
                    list(request.recovery_history),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Previous-command state: "
                + json.dumps(
                    request.history_state,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Observed-outcome state: "
                + json.dumps(
                    request.outcome_state,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Current state: "
                + json.dumps(
                    request.current_state,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Supervised recovery target semantics: "
                + json.dumps(
                    target_semantics,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                (
                    "First assess the previous atomic subtask using only the "
                    "provided past/current evidence, then explain why the "
                    "supervised corrective motion is appropriate. Do not "
                    "repeat a canned success/failure first sentence; the "
                    "caller adds it. Do not claim the corrective action has "
                    "already succeeded. Return strict JSON: "
                    '{"reasoning":"..."}.'
                ),
            ]
        )
        body = post_chat(
            endpoint=self.endpoint,
            payload={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You create evidence-grounded robot recovery reasoning. Separate intended action from observed effect and use no future evidence. Output strict JSON.",
                    },
                    {
                        "role": "user",
                        "content": multimodal_content(
                            text, request.image_paths, request.image_roles
                        ),
                    },
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
            timeout=self.timeout,
            api_key=self.api_key,
        )
        value, _hidden, _finish = _response_parts(body)
        reasoning = str(value.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("recovery reasoner returned empty reasoning")
        return reasoning


def _target_for_segment(
    segment: RecoverySegment,
    states: np.ndarray,
    actions: np.ndarray,
) -> dict[str, Any]:
    start = segment.start_frame
    end = segment.end_frame
    return {
        "guide_action": _round_vector(actions[end] - states[start]),
        "primary_action_verb": segment.primary_action_verb,
        "sub_task": segment.sub_task,
        "num_chunks": int(math.ceil((end - start + 1) / RECOVERY_CHUNK_SIZE)),
        "recovery_role": segment.recovery_role,
    }


def build_recovery_action_episode(
    episode: RecoveryEpisode,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build action supervision without using success-only segmentation."""

    loaded = source or load_recovery_source(episode)
    states = np.asarray(loaded["observation.state"], dtype=np.float64)
    actions = np.asarray(loaded["action"], dtype=np.float64)
    if states.ndim != 2 or actions.shape != states.shape:
        raise ValueError(
            "recovery observation.state/action must have matching [T,D] shape"
        )
    if states.shape[1] != len(episode.state_layout):
        raise ValueError("recovery state layout does not match dataset dimension")
    segments: list[dict[str, Any]] = []
    for segment in episode.segments:
        validate_recovery_segment_rows(loaded, segment)
        target = _target_for_segment(segment, states, actions)
        segments.append(
            {
                "segment_index": segment.segment_index,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "frame_count": segment.end_frame - segment.start_frame + 1,
                "label": segment.label,
                "boundary_source": segment.boundary_source,
                "current_state": _round_vector(states[segment.start_frame]),
                "previous_outcome": asdict(segment.previous_outcome),
                **target,
            }
        )
    record = {
        "schema": RECOVERY_ACTION_EPISODE_SCHEMA,
        "data_kind": "recovery",
        "episode_id": episode.episode_id,
        "episode_index": episode.episode_index,
        "task": episode.task,
        "recovery_level": episode.recovery_level,
        "recovery_level_name": episode.recovery_level_name,
        "failure_type": episode.failure_type,
        "source_failure_verified": episode.source_failure_verified,
        "recovery_success": episode.recovery_success,
        "training_eligible": episode.training_eligible,
        "frame_count": int(len(states)),
        "state_layout": list(episode.state_layout),
        "segmentation": {
            "method": "annotated_recovery_decision_segments",
            "automatic_state_segmentation_is_authoritative": False,
            "policy": (
                "Audited semantic recovery boundaries are authoritative; "
                "kinematic segmentation may only propose boundaries for "
                "later human/collector annotation."
            ),
        },
        "provenance": dict(episode.provenance),
        "segments": segments,
    }
    validate_recovery_action_episode(record)
    return record


def validate_recovery_action_episode(record: Mapping[str, Any]) -> None:
    if record.get("schema") != RECOVERY_ACTION_EPISODE_SCHEMA:
        raise ValueError("unexpected recovery action episode schema")
    if record.get("data_kind") != "recovery":
        raise ValueError("recovery action episode must set data_kind=recovery")
    for field in (
        "source_failure_verified",
        "recovery_success",
        "training_eligible",
    ):
        if record.get(field) is not True:
            raise ValueError(f"recovery action episode requires {field}=true")
    layout = record.get("state_layout")
    segments = record.get("segments")
    if not isinstance(layout, list) or not layout:
        raise ValueError("recovery action episode requires state_layout")
    if not isinstance(segments, list) or not segments:
        raise ValueError("recovery action episode requires segments")
    previous_end: int | None = None
    for index, segment in enumerate(segments):
        if int(segment["segment_index"]) != index:
            raise ValueError("recovery segment indices must be contiguous")
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if previous_end is not None and start != previous_end + 1:
            raise ValueError("recovery segments must be contiguous")
        if end < start:
            raise ValueError("recovery segment must be non-empty")
        previous_end = end
        if len(segment["guide_action"]) != len(layout):
            raise ValueError("recovery guide_action dimension mismatch")
        outcome = segment.get("previous_outcome")
        if not isinstance(outcome, Mapping) or not outcome.get("status"):
            raise ValueError("recovery segment requires previous_outcome")


def generate_recovery_action_dataset(
    episodes: Sequence[RecoveryEpisode],
) -> dict[str, Any]:
    records = [build_recovery_action_episode(episode) for episode in episodes]
    return {
        "schema": RECOVERY_ACTION_DATASET_SCHEMA,
        "data_kind": "recovery",
        "episode_count": len(records),
        "segment_count": sum(len(record["segments"]) for record in records),
        "episodes": records,
    }


def _human_prompt(
    *,
    episode: RecoveryEpisode,
    segment: Mapping[str, Any],
    current_state: Mapping[str, Any],
    history_state: Mapping[str, Any] | None,
    outcome_state: Mapping[str, Any] | None,
    observed_state_delta: Sequence[float] | None,
    recovery_history: Sequence[Mapping[str, Any]],
    image_roles: Sequence[str],
) -> str:
    previous = segment["previous_outcome"]
    previous_command = {
        "primary_action_verb": previous.get(
            "expected_primary_action_verb",
            "",
        ),
        "sub_task": previous.get("expected_subtask", ""),
    }
    lines = [
        "".join("<image>" for _ in image_roles),
        f"Task: {episode.task}",
        "Image roles: " + ", ".join(image_roles),
        "Previous command: "
        + json.dumps(
            previous_command,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "Completed recovery history: "
        + (
            json.dumps(
                list(recovery_history),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if recovery_history
            else "none"
        ),
    ]
    if history_state is not None:
        lines.append(
            "Previous-command state: "
            + json.dumps(
                history_state,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if outcome_state is not None:
        lines.append(
            "Observed-outcome state: "
            + json.dumps(
                outcome_state,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    if observed_state_delta is not None:
        lines.append(
            "Observed state delta across the previous subtask: "
            + json.dumps(
                _round_vector(observed_state_delta),
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
    lines.extend(
        [
            "Current robot state: "
            + json.dumps(
                current_state,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            (
                "First determine whether the previous atomic subtask was "
                "completed, partially completed, failed with no effect, or "
                "produced a different result. Then explain the next atomic "
                "recovery motion in <think>...</think> and output one JSON "
                "action object. Use no future evidence."
            ),
        ]
    )
    return "\n".join(lines)


def _assistant_response(
    reasoning: str,
    target: Mapping[str, Any],
    outcome_status: str,
) -> str:
    prefix = _previous_outcome_prefix(outcome_status)
    detail = _previous_outcome_detail(outcome_status)
    normalized = re.sub(r"\s+", " ", _strip_outcome_prefix(reasoning)).strip()
    if not normalized:
        raise ValueError("recovery reasoning must not be empty")
    if "Therefore, the next action should be:" in normalized:
        raise ValueError(
            "adapter model must not supply the programmatic final action sentence"
        )
    parts = [
        value
        for value in (prefix, detail, normalized, adapter_final_sentence(target))
        if value
    ]
    answer = {
        key: target[key]
        for key in (
            "guide_action",
            "primary_action_verb",
            "sub_task",
            "num_chunks",
        )
    }
    return (
        "<think>\n"
        + "\n".join(parts)
        + "\n</think>\n\n"
        + json.dumps(
            answer,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def _append_frame_images(
    *,
    frame: int | None,
    prefix: str,
    episode: RecoveryEpisode,
    assets: Mapping[tuple[str, int], Path],
    roles: list[str],
    paths: list[Path],
    frames: list[int],
    seen: set[tuple[str, int]],
) -> None:
    if frame is None:
        return
    for camera in episode.camera_keys:
        key = (camera, int(frame))
        if key in seen:
            continue
        seen.add(key)
        roles.append(f"{prefix}_{camera}")
        paths.append(assets[key])
        frames.append(int(frame))


def build_recovery_cot_rows(
    episode: RecoveryEpisode,
    action_episode: Mapping[str, Any],
    *,
    output_dir: Path | str,
    reasoner: RecoveryReasoner,
) -> list[dict[str, Any]]:
    """Build one outcome-aware CoT row per annotated recovery segment."""

    source = load_recovery_source(episode)
    states = np.asarray(source["observation.state"], dtype=np.float64)
    requested_frames: set[int] = set()
    for segment in action_episode["segments"]:
        requested_frames.add(int(segment["start_frame"]))
        previous = segment["previous_outcome"]
        for field in ("history_frame", "outcome_frame"):
            if previous.get(field) is not None:
                requested_frames.add(int(previous[field]))
    videos = {
        camera: recovery_video_path(
            episode.data_root,
            episode.episode_index,
            camera,
        )
        for camera in episode.camera_keys
    }
    asset_root = Path(output_dir) / "assets" / _safe_component(episode.episode_id)
    assets = materialize_camera_frames(
        videos,
        {camera: requested_frames for camera in episode.camera_keys},
        asset_root,
    )
    output: list[dict[str, Any]] = []
    recovery_history: list[dict[str, Any]] = []
    for segment in action_episode["segments"]:
        previous = segment["previous_outcome"]
        current_frame = int(segment["start_frame"])
        history_frame = (
            int(previous["history_frame"])
            if previous.get("history_frame") is not None
            else None
        )
        outcome_frame = (
            int(previous["outcome_frame"])
            if previous.get("outcome_frame") is not None
            else None
        )
        roles: list[str] = []
        image_paths: list[Path] = []
        image_frames: list[int] = []
        seen: set[tuple[str, int]] = set()
        _append_frame_images(
            frame=history_frame,
            prefix="previous_command",
            episode=episode,
            assets=assets,
            roles=roles,
            paths=image_paths,
            frames=image_frames,
            seen=seen,
        )
        if outcome_frame != current_frame:
            _append_frame_images(
                frame=outcome_frame,
                prefix="observed_outcome",
                episode=episode,
                assets=assets,
                roles=roles,
                paths=image_paths,
                frames=image_frames,
                seen=seen,
            )
        _append_frame_images(
            frame=current_frame,
            prefix="current",
            episode=episode,
            assets=assets,
            roles=roles,
            paths=image_paths,
            frames=image_frames,
            seen=seen,
        )
        relative_images = [
            path.relative_to(Path(output_dir)).as_posix() for path in image_paths
        ]
        current_state = _named_state(states[current_frame], episode.state_layout)
        history_state = (
            _named_state(states[history_frame], episode.state_layout)
            if history_frame is not None
            else None
        )
        outcome_state = (
            _named_state(states[outcome_frame], episode.state_layout)
            if outcome_frame is not None
            else None
        )
        observed_delta = (
            states[outcome_frame] - states[history_frame]
            if history_frame is not None and outcome_frame is not None
            else None
        )
        target = {
            key: segment[key]
            for key in (
                "guide_action",
                "primary_action_verb",
                "sub_task",
                "num_chunks",
                "recovery_role",
            )
        }
        sample_id = (
            f"{_safe_component(episode.episode_id)}"
            f"__recovery_segment_{int(segment['segment_index']):03d}"
        )
        reasoning = reasoner.generate(
            RecoveryReasoningRequest(
                sample_id=sample_id,
                task=episode.task,
                recovery_level=episode.recovery_level,
                recovery_level_name=episode.recovery_level_name,
                failure_type=episode.failure_type,
                previous_outcome=previous,
                current_state=current_state,
                history_state=history_state,
                outcome_state=outcome_state,
                recovery_history=tuple(recovery_history),
                target=target,
                image_paths=tuple(image_paths),
                image_roles=tuple(roles),
                style_id=int(segment["segment_index"]) % len(STYLE_SPECS),
            )
        )
        human = _human_prompt(
            episode=episode,
            segment=segment,
            current_state=current_state,
            history_state=history_state,
            outcome_state=outcome_state,
            observed_state_delta=observed_delta,
            recovery_history=recovery_history,
            image_roles=roles,
        )
        row = {
            "schema": RECOVERY_COT_DATASET_SCHEMA,
            "release_eligible": False,
            "data_kind": "recovery",
            "sample_id": sample_id,
            "sample_type": "recovery_transition",
            "episode_id": episode.episode_id,
            "episode_index": episode.episode_index,
            "segment_index": int(segment["segment_index"]),
            "task": episode.task,
            "state_layout": list(episode.state_layout),
            "recovery_level": episode.recovery_level,
            "recovery_level_name": episode.recovery_level_name,
            "failure_type": episode.failure_type,
            "source_failure_verified": True,
            "recovery_success": True,
            "training_eligible": True,
            "images": relative_images,
            "image_roles": roles,
            "image_frames": image_frames,
            "current_frame": current_frame,
            "segment_start_frame": int(segment["start_frame"]),
            "segment_end_frame": int(segment["end_frame"]),
            "boundary_source": segment["boundary_source"],
            "guide_action_semantics": (
                "action[segment_end_frame] - observation.state[segment_start_frame]"
            ),
            "current_state": current_state,
            "previous_command_frame": history_frame,
            "previous_command_state": history_state,
            "previous_outcome_frame": outcome_frame,
            "previous_outcome_state": outcome_state,
            "observed_state_delta": (
                _round_vector(observed_delta) if observed_delta is not None else None
            ),
            "history_outcome": previous["status"],
            "recovery_history": list(recovery_history),
            "previous_outcome": previous,
            "target": target,
            "conversations": [
                {"from": "human", "value": human},
                {
                    "from": "gpt",
                    "value": _assistant_response(
                        reasoning,
                        target,
                        str(previous["status"]),
                    ),
                },
            ],
            "generation": {
                "reasoner": reasoner.name,
                "action_conditioned": True,
                "future_images_visible": False,
                "previous_outcome_is_audited_input": True,
                "previous_outcome_hidden_from_training_human": True,
                "numeric_target_hidden_from_training_human": True,
                "semantic_target_may_equal_previous_command_for_retry": True,
            },
        }
        validate_recovery_cot_row(row)
        output.append(row)
        recovery_history.append(
            {
                "segment_index": int(segment["segment_index"]),
                "sub_task": str(segment["sub_task"]),
                "primary_action_verb": str(segment["primary_action_verb"]),
            }
        )
    return output


def validate_recovery_cot_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != RECOVERY_COT_DATASET_SCHEMA:
        raise ValueError("unexpected recovery CoT row schema")
    if row.get("data_kind") != "recovery":
        raise ValueError("recovery CoT row must set data_kind=recovery")
    for field in (
        "source_failure_verified",
        "recovery_success",
        "training_eligible",
    ):
        if row.get(field) is not True:
            raise ValueError(f"recovery CoT requires {field}=true")
    previous = row.get("previous_outcome")
    if not isinstance(previous, Mapping):
        raise ValueError("recovery CoT requires previous_outcome")
    if row.get("history_outcome") != previous.get("status"):
        raise ValueError("history_outcome must match previous_outcome.status")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("recovery CoT row requires one human/gpt pair")
    if conversations[0].get("from") != "human":
        raise ValueError("recovery CoT first conversation must be human")
    if conversations[1].get("from") != "gpt":
        raise ValueError("recovery CoT second conversation must be gpt")
    human = str(conversations[0].get("value", ""))
    assistant = str(conversations[1].get("value", ""))
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("recovery CoT row requires images")
    if human.count("<image>") != len(images):
        raise ValueError("recovery image placeholders do not match image paths")
    match = re.fullmatch(
        r"\s*<think>\s*(.+?)\s*</think>\s*\{.*\}\s*",
        assistant,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("recovery assistant must embed CoT and action JSON")
    think = match.group(1).strip()
    expected_prefix = _previous_outcome_prefix(str(previous.get("status")))
    if expected_prefix and not think.startswith(expected_prefix):
        raise ValueError(
            "recovery CoT does not begin with the required previous-outcome assessment"
        )
    if previous.get("status") != "success" and think.startswith(SUCCESS_PREFIX):
        raise ValueError("non-success previous outcome uses success prefix")
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("recovery CoT requires target metadata")
    answer_text = assistant.split("</think>", 1)[1].strip()
    try:
        answer = json.loads(answer_text)
    except json.JSONDecodeError as exc:
        raise ValueError("recovery assistant action JSON is invalid") from exc
    expected_answer = {
        key: target[key]
        for key in (
            "guide_action",
            "primary_action_verb",
            "sub_task",
            "num_chunks",
        )
    }
    if answer != expected_answer:
        raise ValueError("recovery assistant action JSON differs from target")
    if not think.endswith(adapter_final_sentence(target)):
        raise ValueError("recovery adapter final action sentence differs from target")
    guide = json.dumps(target["guide_action"], separators=(",", ":"))
    if guide in human:
        raise ValueError("recovery training human exposes numeric guide_action")
    for marker in (
        '"status":',
        '"observed_outcome":',
        '"mismatch":',
        '"annotation_source":',
    ):
        if marker in human:
            raise ValueError(
                "recovery training human exposes the audited previous-outcome label"
            )


def generate_recovery_cot_dataset(
    episodes: Sequence[RecoveryEpisode],
    output_dir: Path | str,
    reasoner: RecoveryReasoner,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    action_records: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        action_episode = build_recovery_action_episode(episode)
        action_records.append(action_episode)
        rows.extend(
            build_recovery_cot_rows(
                episode,
                action_episode,
                output_dir=output_dir,
                reasoner=reasoner,
            )
        )
    action_dataset = {
        "schema": RECOVERY_ACTION_DATASET_SCHEMA,
        "data_kind": "recovery",
        "episode_count": len(action_records),
        "segment_count": sum(len(record["segments"]) for record in action_records),
        "episodes": action_records,
    }
    return action_dataset, rows


def recovery_generation_manifest(
    *,
    source_manifest: Path | str,
    episodes: Sequence[RecoveryEpisode],
    action_dataset: Mapping[str, Any],
    cot_rows: Sequence[Mapping[str, Any]],
    reasoner: RecoveryReasoner,
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in cot_rows:
        status = str(row["previous_outcome"]["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema": "atomic_recovery_generation_manifest",
        "release_eligible": False,
        "release_gate": "experimental arbitrary-layout adapter; use native recovery_cot_data for formal planner data",
        "data_kind": "recovery",
        "source_manifest": str(Path(source_manifest).resolve()),
        "episode_count": len(episodes),
        "segment_count": int(action_dataset["segment_count"]),
        "cot_row_count": len(cot_rows),
        "reasoner": reasoner.name,
        "previous_outcome_counts": status_counts,
        "segmentation_policy": {
            "success_pipeline_unchanged": True,
            "automatic_state_segmentation_is_authoritative": False,
            "authoritative_source": "audited_semantic_recovery_segments",
        },
        "episodes": [
            {
                "episode_id": episode.episode_id,
                "episode_index": episode.episode_index,
                "data_root": str(episode.data_root),
                "recovery_level": episode.recovery_level,
                "failure_type": episode.failure_type,
                "segment_count": len(episode.segments),
            }
            for episode in episodes
        ],
    }
