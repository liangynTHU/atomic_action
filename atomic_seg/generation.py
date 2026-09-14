"""Action-only and CoT dataset generation for successful episodes."""

from __future__ import annotations

import base64
import json
import math
import mimetypes
import re
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from .config import SegmentationConfig
from .io import (
    CAMERA_KEYS,
    SuccessEpisode,
    load_episode,
    resolve_episode_tasks,
    video_path,
)
from .media import materialize_camera_frames
from .segmentation import segment_episode, serializable_segmentation


ACTION_DATASET_SCHEMA = "atomic_action_dataset"
ACTION_EPISODE_SCHEMA = "atomic_action_episode"
COT_DATASET_SCHEMA = "atomic_cot_training_row"
CHUNK_SIZE = 8
FORBIDDEN_COT_KEYS = {
    "cot",
    "chain_of_thought",
    "reasoning",
    "reasoning_steps",
    "assistant_response",
}


def _round_vector(value: Sequence[Any]) -> list[float]:
    output: list[float] = []
    for item in value:
        number = round(float(item), 4)
        output.append(0.0 if number == 0.0 else number)
    return output


def semantic_action(
    action: str,
    *,
    gripper_decrease_means_close: bool,
) -> str:
    if action == "gripper_decrease":
        return (
            "close_gripper"
            if gripper_decrease_means_close
            else "open_gripper"
        )
    if action == "gripper_increase":
        return (
            "open_gripper"
            if gripper_decrease_means_close
            else "close_gripper"
        )
    return action


def primary_action_verb(left_action: str, right_action: str) -> str:
    return f"left:{left_action}, right:{right_action}"


def _arm_clause(arm: str, action: str) -> str:
    if action == "still":
        return f"keep the {arm} arm still"
    if action == "move":
        return f"move the {arm} arm"
    if action == "turn":
        return f"turn the {arm} end effector"
    if action == "lift":
        return f"lift the {arm} arm"
    if action == "lower":
        return f"lower the {arm} arm"
    if action == "close_gripper":
        return f"close the {arm} gripper"
    if action == "open_gripper":
        return f"open the {arm} gripper"
    raise ValueError(f"unsupported action {action!r}")


def action_description(
    left_action: str,
    right_action: str,
    task: str,
) -> str:
    """Build a conservative, action-only sub-task description."""

    if left_action == right_action == "still":
        command = "keep both arms still"
    elif left_action == "still":
        command = (
            f"{_arm_clause('right', right_action)} while keeping "
            "the left arm still"
        )
    elif right_action == "still":
        command = (
            f"{_arm_clause('left', left_action)} while keeping "
            "the right arm still"
        )
    else:
        command = (
            f"{_arm_clause('left', left_action)} and "
            f"{_arm_clause('right', right_action)}"
        )
    return f"To continue the task '{task}', {command}."


def _named_state(value: Sequence[Any]) -> dict[str, Any]:
    values = _round_vector(value)
    if len(values) != 16:
        raise ValueError("robot state must contain 16 values")
    return {
        "left": {
            "xyz": values[0:3],
            "quaternion_wxyz": values[3:7],
            "gripper": values[7],
        },
        "right": {
            "xyz": values[8:11],
            "quaternion_wxyz": values[11:15],
            "gripper": values[15],
        },
    }


def _action_target(
    segment: Mapping[str, Any],
    states: np.ndarray,
    actions: np.ndarray,
    task: str,
    config: SegmentationConfig,
) -> dict[str, Any]:
    start = int(segment["start_frame"])
    end = int(segment["end_frame"])
    if end >= len(actions):
        raise ValueError(f"segment endpoint {end} has no action row")
    left = semantic_action(
        str(segment["left_action"]),
        gripper_decrease_means_close=(
            config.gripper_decrease_means_close
        ),
    )
    right = semantic_action(
        str(segment["right_action"]),
        gripper_decrease_means_close=(
            config.gripper_decrease_means_close
        ),
    )
    return {
        "primary_action_verb": primary_action_verb(left, right),
        "left_action": left,
        "right_action": right,
        "sub_task": action_description(left, right, task),
        "guide_action": _round_vector(actions[end] - states[start]),
        "num_chunks": int(math.ceil((end - start + 1) / CHUNK_SIZE)),
    }


def build_action_episode(
    episode: SuccessEpisode,
    source: Mapping[str, Any],
    *,
    config: SegmentationConfig | None = None,
) -> dict[str, Any]:
    """Create one action-only episode record."""

    selected_config = config or SegmentationConfig()
    states = np.asarray(source["observation.state"], dtype=np.float64)
    actions = np.asarray(source["action"], dtype=np.float64)
    timestamps = np.asarray(source["timestamp"], dtype=np.float64)
    if actions.shape != states.shape:
        raise ValueError(
            "action and observation.state must share the same shape"
        )
    segmentation = segment_episode(states, timestamps, selected_config)
    segments: list[dict[str, Any]] = []
    for segment in segmentation["segments"]:
        target = _action_target(
            segment,
            states,
            actions,
            episode.task,
            selected_config,
        )
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        segments.append(
            {
                "segment_index": int(segment["segment_index"]),
                "start_frame": start,
                "end_frame": end,
                "frame_count": end - start + 1,
                "coordination_relation": segment[
                    "coordination_relation"
                ],
                "current_state": _round_vector(states[start]),
                **target,
            }
        )
    record = {
        "schema": ACTION_EPISODE_SCHEMA,
        "episode_id": (
            episode.source_episode_id
            or f"episode_{episode.episode_index:06d}"
        ),
        "episode_index": episode.episode_index,
        "task": episode.task,
        "success": True,
        "success_provenance": episode.success_provenance,
        "frame_count": int(len(states)),
        "state_layout": segmentation["state_layout"],
        "segments": segments,
        "segmentation": {
            key: value
            for key, value in serializable_segmentation(
                segmentation
            ).items()
            if key
            in {
                "method",
                "decision_input_policy",
                "temporal_parameters",
                "boundaries",
                "per_arm_boundaries",
                "boundary_evidence",
            }
        },
    }
    validate_action_episode(record)
    return record


def generate_action_dataset(
    data_root: Path | str,
    episodes: Sequence[SuccessEpisode],
    *,
    config: SegmentationConfig | None = None,
) -> dict[str, Any]:
    """Generate an action-only dataset from curated successful episodes."""

    resolved = resolve_episode_tasks(data_root, episodes)
    records = [
        build_action_episode(
            episode,
            load_episode(data_root, episode.episode_index),
            config=config,
        )
        for episode in resolved
    ]
    return {
        "schema": ACTION_DATASET_SCHEMA,
        "success_only": True,
        "episode_count": len(records),
        "segment_count": sum(
            len(record["segments"]) for record in records
        ),
        "episodes": records,
    }


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_keys(child)


def validate_action_episode(record: Mapping[str, Any]) -> None:
    if record.get("schema") != ACTION_EPISODE_SCHEMA:
        raise ValueError("unexpected action episode schema")
    if record.get("success") is not True:
        raise ValueError("action records must be successful demonstrations")
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("action episode must contain segments")
    forbidden = FORBIDDEN_COT_KEYS & set(_walk_keys(record))
    if forbidden:
        raise ValueError(
            f"action-only data contains CoT fields: {sorted(forbidden)}"
        )
    previous_end = -1
    for expected_index, segment in enumerate(segments):
        if int(segment["segment_index"]) != expected_index:
            raise ValueError("segment indices must be contiguous")
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        if start != previous_end + 1 or end < start:
            raise ValueError("segments must be contiguous and non-empty")
        previous_end = end
        if len(segment["guide_action"]) != 16:
            raise ValueError("guide_action must contain 16 values")


@dataclass(frozen=True)
class ReasoningRequest:
    sample_id: str
    task: str
    current_state: Mapping[str, Any]
    history_actions: Sequence[str]
    previous_state: Mapping[str, Any] | None
    target: Mapping[str, Any]
    image_paths: Sequence[Path]
    image_roles: Sequence[str]


class Reasoner(Protocol):
    name: str

    def generate(self, request: ReasoningRequest) -> str:
        """Return public pre-action reasoning without a final action."""


class TemplateReasoner:
    """Offline action-conditioned baseline for tests and examples."""

    name = "deterministic_action_conditioned_template"

    def generate(self, request: ReasoningRequest) -> str:
        history = (
            "No atomic action has been completed yet."
            if not request.history_actions
            else (
                f"{len(request.history_actions)} atomic actions have already "
                "been completed, so the episode is at a later decision point."
            )
        )
        command = str(request.target["sub_task"])
        return (
            f"The overall objective is {request.task.rstrip('.')}. "
            f"{history} The current observation and robot state define the "
            f"next pre-action decision. {command} This statement describes "
            "the commanded motion only and does not claim contact, grasp, "
            "placement, or task success."
        )


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


class OpenAICompatibleReasoner:
    """Multimodal reasoner backed by an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_tokens: int = 700,
        temperature: float = 0.2,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.name = f"openai_compatible:{model}"

    def generate(self, request: ReasoningRequest) -> str:
        target = {
            key: request.target[key]
            for key in (
                "primary_action_verb",
                "left_action",
                "right_action",
                "sub_task",
            )
        }
        text = "\n".join(
            [
                f"Task: {request.task}",
                "Completed actions: "
                + (
                    json.dumps(
                        list(request.history_actions),
                        ensure_ascii=False,
                    )
                    if request.history_actions
                    else "none"
                ),
                "Current state: "
                + json.dumps(
                    request.current_state,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "Previous state: "
                + (
                    json.dumps(
                        request.previous_state,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    if request.previous_state is not None
                    else "none"
                ),
                "Supervised atomic target: "
                + json.dumps(
                    target,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                (
                    "Write a concise pre-action explanation grounded only in "
                    "the current/history images and states. Explain why this "
                    "atomic motion is appropriate for continuing the task. "
                    "Do not cite the label or target as justification; do not "
                    "claim contact, grasp, placement, or success unless it is "
                    "already visible. Return JSON: {\"reasoning\":\"...\"}."
                ),
            ]
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for role, path in zip(
            request.image_roles,
            request.image_paths,
        ):
            content.append(
                {"type": "text", "text": f"Image role: {role}"}
            )
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _data_url(path)},
                }
            )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You create supervised pre-action robot reasoning. "
                        "Use only current and past evidence. Output strict JSON."
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        http_request = urllib.request.Request(
            f"{self.endpoint}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(
            http_request,
            timeout=self.timeout,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
        content_value = body["choices"][0]["message"]["content"]
        if isinstance(content_value, list):
            content_value = "".join(
                str(item.get("text", ""))
                for item in content_value
                if isinstance(item, dict)
            )
        match = re.search(r"\{.*\}", str(content_value), re.DOTALL)
        if match is None:
            raise ValueError("reasoner response does not contain JSON")
        value = json.loads(match.group(0))
        reasoning = str(value.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("reasoner returned empty reasoning")
        return reasoning


def _human_prompt(
    *,
    task: str,
    current_frame: int,
    current_state: Mapping[str, Any],
    history_actions: Sequence[str],
    previous_frame: int | None,
    previous_state: Mapping[str, Any] | None,
    image_roles: Sequence[str],
) -> str:
    lines = [
        "".join("<image>" for _ in image_roles),
        f"Task: {task}",
        f"Current decision frame: {current_frame}",
        "Image roles: " + ", ".join(image_roles),
        "Completed action history: "
        + (
            json.dumps(
                list(history_actions),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if history_actions
            else "none"
        ),
        "Current robot state: "
        + json.dumps(
            current_state,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    ]
    if previous_frame is not None and previous_state is not None:
        lines.extend(
            [
                f"Previous decision frame: {previous_frame}",
                "Previous robot state: "
                + json.dumps(
                    previous_state,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )
    lines.append(
        "Based only on the current and completed-history evidence, explain "
        "the next atomic motion in <think>...</think>, then output one JSON "
        "action object."
    )
    return "\n".join(lines)


def _assistant_response(
    reasoning: str,
    target: Mapping[str, Any],
) -> str:
    normalized = re.sub(r"\s+", " ", reasoning).strip()
    if not normalized:
        raise ValueError("reasoning must not be empty")
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
        f"<think>\n{normalized}\n</think>\n\n"
        + json.dumps(
            answer,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def build_cot_rows(
    action_episode: Mapping[str, Any],
    *,
    data_root: Path | str,
    output_dir: Path | str,
    reasoner: Reasoner,
) -> list[dict[str, Any]]:
    """Build one CoT training row per atomic segment."""

    episode_index = int(action_episode["episode_index"])
    source = load_episode(data_root, episode_index)
    states = np.asarray(source["observation.state"], dtype=np.float64)
    segments = list(action_episode["segments"])
    requested_frames = {
        int(segment["start_frame"]) for segment in segments
    }
    requested_frames.update(
        int(segments[index - 1]["start_frame"])
        for index in range(1, len(segments))
    )
    videos = {
        camera: video_path(data_root, episode_index, camera)
        for camera in CAMERA_KEYS
    }
    asset_root = (
        Path(output_dir)
        / "assets"
        / f"episode_{episode_index:06d}"
    )
    assets = materialize_camera_frames(
        videos,
        {camera: requested_frames for camera in CAMERA_KEYS},
        asset_root,
    )
    output: list[dict[str, Any]] = []
    history_actions: list[str] = []
    for index, segment in enumerate(segments):
        current_frame = int(segment["start_frame"])
        previous_frame = (
            int(segments[index - 1]["start_frame"])
            if index > 0
            else None
        )
        roles: list[str] = []
        image_paths: list[Path] = []
        image_frames: list[int] = []
        if previous_frame is not None:
            for camera in CAMERA_KEYS:
                roles.append(f"history_{camera}")
                image_paths.append(assets[(camera, previous_frame)])
                image_frames.append(previous_frame)
        for camera in CAMERA_KEYS:
            roles.append(f"current_{camera}")
            image_paths.append(assets[(camera, current_frame)])
            image_frames.append(current_frame)
        relative_images = [
            path.relative_to(Path(output_dir)).as_posix()
            for path in image_paths
        ]
        current_state = _named_state(states[current_frame])
        previous_state = (
            _named_state(states[previous_frame])
            if previous_frame is not None
            else None
        )
        target = {
            key: segment[key]
            for key in (
                "guide_action",
                "primary_action_verb",
                "left_action",
                "right_action",
                "sub_task",
                "num_chunks",
            )
        }
        sample_id = (
            f"episode_{episode_index:06d}"
            f"__segment_{int(segment['segment_index']):03d}"
        )
        reasoning = reasoner.generate(
            ReasoningRequest(
                sample_id=sample_id,
                task=str(action_episode["task"]),
                current_state=current_state,
                history_actions=tuple(history_actions),
                previous_state=previous_state,
                target=target,
                image_paths=tuple(image_paths),
                image_roles=tuple(roles),
            )
        )
        human = _human_prompt(
            task=str(action_episode["task"]),
            current_frame=current_frame,
            current_state=current_state,
            history_actions=history_actions,
            previous_frame=previous_frame,
            previous_state=previous_state,
            image_roles=roles,
        )
        row = {
            "schema": COT_DATASET_SCHEMA,
            "sample_id": sample_id,
            "episode_id": action_episode["episode_id"],
            "episode_index": episode_index,
            "segment_index": int(segment["segment_index"]),
            "task": action_episode["task"],
            "success": True,
            "images": relative_images,
            "image_roles": roles,
            "image_frames": image_frames,
            "current_frame": current_frame,
            "history_actions": list(history_actions),
            "target": target,
            "conversations": [
                {"from": "human", "value": human},
                {
                    "from": "gpt",
                    "value": _assistant_response(reasoning, target),
                },
            ],
            "generation": {
                "reasoner": reasoner.name,
                "action_conditioned": True,
                "future_images_visible": False,
                "target_hidden_from_training_human": True,
            },
        }
        validate_cot_row(row)
        output.append(row)
        history_actions.append(str(segment["primary_action_verb"]))
    return output


def generate_cot_dataset(
    data_root: Path | str,
    episodes: Sequence[SuccessEpisode],
    output_dir: Path | str,
    reasoner: Reasoner,
    *,
    config: SegmentationConfig | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate aligned action-only episodes and CoT rows."""

    action_dataset = generate_action_dataset(
        data_root,
        episodes,
        config=config,
    )
    rows: list[dict[str, Any]] = []
    for episode in action_dataset["episodes"]:
        rows.extend(
            build_cot_rows(
                episode,
                data_root=data_root,
                output_dir=output_dir,
                reasoner=reasoner,
            )
        )
    return action_dataset, rows


def validate_cot_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != COT_DATASET_SCHEMA:
        raise ValueError("unexpected CoT row schema")
    if row.get("success") is not True:
        raise ValueError("CoT rows must be successful demonstrations")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError("CoT row must contain one human/gpt pair")
    human = str(conversations[0].get("value", ""))
    assistant = str(conversations[1].get("value", ""))
    if conversations[0].get("from") != "human":
        raise ValueError("first conversation must be human")
    if conversations[1].get("from") != "gpt":
        raise ValueError("second conversation must be gpt")
    images = row.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("CoT row must contain images")
    if human.count("<image>") != len(images):
        raise ValueError("image placeholders do not match image paths")
    if not re.fullmatch(
        r"\s*<think>\s*.+?\s*</think>\s*\{.*\}\s*",
        assistant,
        re.DOTALL,
    ):
        raise ValueError("assistant must embed CoT followed by action JSON")
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("CoT row target metadata is required")
    for value in (
        json.dumps(target["guide_action"], separators=(",", ":")),
        str(target["sub_task"]),
    ):
        if value and value in human:
            raise ValueError("training human exposes the current target")


def generation_manifest(
    *,
    data_root: Path | str,
    episodes: Sequence[SuccessEpisode],
    action_dataset: Mapping[str, Any],
    cot_rows: Sequence[Mapping[str, Any]] | None = None,
    reasoner: Reasoner | None = None,
    config: SegmentationConfig | None = None,
) -> dict[str, Any]:
    return {
        "schema": "atomic_generation_manifest",
        "data_root": str(Path(data_root).resolve()),
        "success_only": True,
        "episodes": [asdict(episode) for episode in episodes],
        "episode_count": int(action_dataset["episode_count"]),
        "segment_count": int(action_dataset["segment_count"]),
        "cot_row_count": len(cot_rows or []),
        "reasoner": reasoner.name if reasoner else None,
        "segmentation_config": asdict(config or SegmentationConfig()),
    }
