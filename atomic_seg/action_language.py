"""Deterministic language for dual-arm action supervision."""

from __future__ import annotations

SUPPORTED_ACTIONS = {
    "still",
    "move",
    "turn",
    "lift",
    "lower",
    "open_gripper",
    "close_gripper",
    "oscillate",
}


def _active_clause(arm: str, action: str) -> str:
    if arm not in {"left", "right"}:
        raise ValueError(f"unsupported arm {arm!r}")
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported action {action!r}")
    return {
        "still": f"keep the {arm} arm stationary",
        "move": f"move the {arm} arm to the demonstrated target pose",
        "turn": (
            f"rotate the {arm} end effector to the demonstrated target orientation"
        ),
        "lift": f"lift the {arm} arm toward the demonstrated target pose",
        "lower": f"lower the {arm} arm toward the demonstrated target pose",
        "open_gripper": f"open the {arm} gripper",
        "close_gripper": f"close the {arm} gripper",
        "oscillate": f"oscillate the {arm} arm along the demonstrated motion",
    }[action]


def canonical_action_clause(left_action: str, right_action: str) -> str:
    """Describe only the commanded actuators, without object-state claims."""

    left = str(left_action)
    right = str(right_action)
    if left not in SUPPORTED_ACTIONS or right not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported action pair: {left!r}, {right!r}")
    if left == right == "still":
        return "keep both arms stationary at the demonstrated terminal pose"
    if left == "still":
        return f"{_active_clause('right', right)} while keeping the left arm stationary"
    if right == "still":
        return f"{_active_clause('left', left)} while keeping the right arm stationary"
    return f"{_active_clause('left', left)} and {_active_clause('right', right)}"


def canonical_sub_task(left_action: str, right_action: str) -> str:
    clause = canonical_action_clause(left_action, right_action)
    return clause[0].upper() + clause[1:] + "."


def cot_final_action_sentence(
    left_action: str,
    right_action: str,
    num_chunks: int,
) -> str:
    chunks = int(num_chunks)
    if chunks < 1:
        raise ValueError("num_chunks must be positive")
    verb = f"left:{left_action}, right:{right_action}"
    clause = canonical_action_clause(left_action, right_action)
    unit = "action chunk" if chunks == 1 else "action chunks"
    return (
        f"Therefore, the next action should be: {clause}, executed for "
        f"{chunks} {unit} ({verb})."
    )
