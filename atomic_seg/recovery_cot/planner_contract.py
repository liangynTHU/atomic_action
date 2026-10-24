"""Local training prompt contract around the legacy Embodied row serializer.

Only this freshly loaded module instance is adapted; no external file is edited.
Its geometry, image, mask and history validators remain active.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from atomic_seg.common.cot_format import split_assistant
from atomic_seg.prompt_contract import (
    MULTI_ARM_SYSTEM_PROMPT,
    SINGLE_ARM_SYSTEM_PROMPT,
    THINKING_USER_ENDING,
    system_prompt_for_mode,
)

from .quality import reasoning_body

LEGACY_USER_ENDING = (
    "Predict the next guide_action, primary_action_verb, sub_task, and "
    "num_chunks as a JSON object."
)


def bind_planner_contract(module: Any) -> Any:
    original = module.build_embodied_user_prompt

    def build_user(**kwargs: Any) -> str:
        prompt = original(**kwargs)
        if not prompt.endswith(LEGACY_USER_ENDING):
            raise ValueError(
                "unrecognized reference-driver user format; review adapter"
            )
        return prompt[: -len(LEGACY_USER_ENDING)] + THINKING_USER_ENDING

    module.ROBOTWIN_VLM_HISTORY_SYSTEM_PROMPT = SINGLE_ARM_SYSTEM_PROMPT
    module.ROBOTWIN_VLM_ALL_TASK_HISTORY_SYSTEM_PROMPT = MULTI_ARM_SYSTEM_PROMPT
    module.build_embodied_user_prompt = build_user
    if hasattr(module, "_embodied_arm_contract"):
        original_arm_contract = module._embodied_arm_contract

        def recovery_arm_contract(task_instruction, source_steps):
            mode = module.episode_arm_mode(source_steps)
            instruction_arm = module.explicit_task_arm(task_instruction)
            if (
                mode in {"left", "right"}
                and instruction_arm in {"left", "right"}
                and mode != instruction_arm
            ):
                return None, None, "audited_recovery_arm_handoff", "multi"
            if mode not in {"left", "right", "bimanual"}:
                return None, None, "terminal_stationary_recovery", "multi"
            return original_arm_contract(task_instruction, source_steps)

        module._embodied_arm_contract = recovery_arm_contract
    return module


def validate_planner_row(row: Mapping[str, Any]) -> None:
    messages = row["messages"]
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("recovery messages must be system/user/assistant")
    if messages[0]["content"] != system_prompt_for_mode(str(row["task_arm_mode"])):
        raise ValueError("recovery system prompt mismatch")
    if not messages[1]["content"].endswith(THINKING_USER_ENDING):
        raise ValueError("recovery thinking user ending mismatch")
    reasoning, answer = split_assistant(messages[2]["content"])
    if json.loads(answer) != row["target"]:
        raise ValueError("recovery assistant JSON differs from target")
    reasoning_body(reasoning, row["target"])
