"""Read and rebuild aligned normal-success VLM/VLA records."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from atomic_seg.common.artifacts import compact_json
from atomic_seg.prompt_contract import NO_THINKING_USER_ENDING, THINKING_USER_ENDING

CHUNK_SIZE = 8
STATE_DIMENSION = 16


def round_prompt(value: Any) -> float:
    rounded = round(float(value), 4)
    return 0.0 if rounded == 0.0 else rounded


def delta(
    before: Sequence[Any],
    after: Sequence[Any],
    *,
    rounded: bool,
) -> list[float]:
    if len(before) != STATE_DIMENSION or len(after) != STATE_DIMENSION:
        raise ValueError("state delta requires two 16-D vectors")
    values = [float(target) - float(origin) for origin, target in zip(before, after)]
    return [round(item, 4) for item in values] if rounded else values


def chunks(action_rows: int) -> int:
    if action_rows <= 0:
        raise ValueError("action row count must be positive")
    return int(math.ceil(action_rows / CHUNK_SIZE))


def named_state(value: Sequence[Any]) -> str:
    values = [float(item) for item in value]
    if len(values) != STATE_DIMENSION:
        raise ValueError("prompt state must be 16-D")
    return json.dumps(
        {
            "left": {
                "xyz": [round_prompt(item) for item in values[0:3]],
                "quat_wxyz": [round_prompt(item) for item in values[3:7]],
                "gripper": round_prompt(values[7]),
            },
            "right": {
                "xyz": [round_prompt(item) for item in values[8:11]],
                "quat_wxyz": [round_prompt(item) for item in values[11:15]],
                "gripper": round_prompt(values[15]),
            },
        },
        ensure_ascii=False,
    )


def named_delta(value: Sequence[Any]) -> str:
    values = [float(item) for item in value]
    if len(values) != STATE_DIMENSION:
        raise ValueError("prompt state delta must be 16-D")
    return json.dumps(
        {
            "left": {
                "delta_xyz": [round_prompt(item) for item in values[0:3]],
                "delta_quat_wxyz": [round_prompt(item) for item in values[3:7]],
                "delta_gripper": round_prompt(values[7]),
            },
            "right": {
                "delta_xyz": [round_prompt(item) for item in values[8:11]],
                "delta_quat_wxyz": [round_prompt(item) for item in values[11:15]],
                "delta_gripper": round_prompt(values[15]),
            },
        },
        ensure_ascii=False,
    )


def canonical_goal(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "guide_action": [round_prompt(item) for item in value["guide_action"]],
        "primary_action_verb": str(value["primary_action_verb"]).strip(),
        "sub_task": str(value["sub_task"]).strip(),
        "num_chunks": int(value["num_chunks"]),
    }


def user_prompt(
    *,
    task: str,
    current_state: Sequence[Any],
    image_count: int,
    prompt_required_arm: str | None,
    task_arm_mode: str,
    previous_state: Sequence[Any] | None,
    previous_goal: Mapping[str, Any] | None,
    executed_chunks: int,
    executed_actions: int,
    source_action_rows: int | None = None,
    padded_action_slots: int | None = None,
    thinking: bool = True,
) -> str:
    lines = [f"Task: {str(task).strip()}"]
    if task_arm_mode == "multi":
        lines.append(
            "Task arm mode: multi-arm; both arms may participate or hand off "
            "as required by the task."
        )
    if prompt_required_arm is not None:
        lines.append(f"Required task arm: {prompt_required_arm}")
    if previous_state is None:
        if previous_goal is not None or image_count != 3:
            raise ValueError("initial row history contract is invalid")
        lines.extend(
            [
                "Images: current head, current left wrist, current right wrist.",
                "History boundary: none; this is the initial planning boundary.",
                "Current state: " + named_state(current_state),
            ]
        )
    else:
        if previous_goal is None or image_count != 6:
            raise ValueError("history row contract is invalid")
        lines.extend(
            [
                (
                    "Images: previous-boundary head, previous-boundary left "
                    "wrist, previous-boundary right wrist, current head, "
                    "current left wrist, current right wrist."
                ),
                "Previous command: "
                + json.dumps(canonical_goal(previous_goal), ensure_ascii=False),
                "Previous execution: "
                + json.dumps(
                    {
                        "executed_chunks": int(executed_chunks),
                        "valid_action_rows": int(
                            executed_actions
                            if source_action_rows is None
                            else source_action_rows
                        ),
                        "aligned_action_slots": int(executed_actions),
                        "padded_action_slots": int(
                            0 if padded_action_slots is None else padded_action_slots
                        ),
                    }
                ),
                "Previous boundary state: " + named_state(previous_state),
                "Current state: " + named_state(current_state),
                "Observed state delta since previous boundary: "
                + named_delta(delta(previous_state, current_state, rounded=False)),
            ]
        )
    lines.append(THINKING_USER_ENDING if thinking else NO_THINKING_USER_ENDING)
    return "<image>" * image_count + "\n".join(lines)


def reasoning_from_assistant(content: str) -> str:
    prefix = "<think>\n"
    delimiter = "\n</think>\n\n"
    if not content.startswith(prefix) or content.count(delimiter) != 1:
        raise ValueError("unexpected VLM assistant format")
    return content[len(prefix) :].split(delimiter, 1)[0].strip()


def assistant(reasoning: str, target: Mapping[str, Any]) -> str:
    return f"<think>\n{str(reasoning).strip()}\n</think>\n\n" + json.dumps(
        target, ensure_ascii=False
    )


def current_images(row: Mapping[str, Any]) -> list[str]:
    images = [str(item) for item in row["images"]]
    if len(images) not in {3, 6}:
        raise ValueError("VLM row must have three or six images")
    return images[-3:]


def sample_id(value: str, segment_index: int) -> str:
    updated, replacements = re.subn(
        r"__segment_\d+$",
        f"__segment_{segment_index:03d}",
        str(value),
    )
    if replacements != 1:
        raise ValueError(f"unexpected sample_id: {value}")
    return updated


def iter_vla(path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            value = line.strip()
            if not value or value in {"{", "}"}:
                continue
            if value.endswith(","):
                value = value[:-1]
            key_text, separator, payload = value.partition(":")
            if not separator:
                raise ValueError(f"{path}:{line_number}: invalid VLA entry")
            key = json.loads(key_text)
            episode = json.loads(payload)
            if not isinstance(key, str) or not isinstance(episode, dict):
                raise ValueError(f"{path}:{line_number}: invalid VLA object")
            yield key, episode


def iter_vlm(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def episode_vlm_rows(
    iterator: Iterator[dict[str, Any]],
    pending: dict[str, Any] | None,
    episode_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    row = pending
    while row is not None and int(row["episode_index"]) < episode_index:
        raise ValueError("VLM contains an episode absent from VLA")
    while row is not None and int(row["episode_index"]) == episode_index:
        rows.append(row)
        row = next(iterator, None)
    return rows, row


_compact_json = compact_json
