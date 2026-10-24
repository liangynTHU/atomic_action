"""Native recovery teacher prompts and candidate/critic validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from atomic_seg.common.artifacts import compact_json
from atomic_seg.common.numeric import round_vector as _round_vector

from .contracts import CAMERAS
from .native import _generic_subtask
from atomic_seg.cot_data.styles import STYLE_SPECS
from .quality import complete_reasoning, validate_body


def image_directory(output_root: Path, record: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "assets/pre_action_v28"
        / str(record["source_variant"])
        / str(record["task_name"])
        / f"episode_{int(record['official_episode_index']):06d}"
    )


def _image_rows(
    output_root: Path,
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    previous_step: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    directory = image_directory(output_root, record)
    rows: list[dict[str, Any]] = []
    if previous_step is not None:
        frame = int(previous_step["start_frame"])
        for camera, _key in CAMERAS:
            rows.append(
                {
                    "evidence_id": f"pre.history.{camera}.f{frame:06d}",
                    "role": f"history_{camera}",
                    "frame": frame,
                    "visibility": "pre_action",
                    "path": str(
                        (directory / f"{camera}_f{frame:06d}.jpg").relative_to(
                            output_root
                        )
                    ),
                }
            )
    frame = int(step["start_frame"])
    for camera, _key in CAMERAS:
        rows.append(
            {
                "evidence_id": f"pre.current.{camera}.f{frame:06d}",
                "role": f"current_{camera}",
                "frame": frame,
                "visibility": "pre_action",
                "path": str(
                    (directory / f"{camera}_f{frame:06d}.jpg").relative_to(output_root)
                ),
            }
        )
    for row in rows:
        path = output_root / str(row["path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return rows


def _named_state(value: Sequence[Any]) -> dict[str, Any]:
    state = _round_vector(value)
    return {
        "left": {
            "xyz": state[0:3],
            "quat_wxyz": state[3:7],
            "gripper": state[7],
        },
        "right": {
            "xyz": state[8:11],
            "quat_wxyz": state[11:15],
            "gripper": state[15],
        },
    }


def _instruction_arm(instruction: str) -> str | None:
    text = " ".join(instruction.lower().replace("-", " ").split())
    left = bool(re.search(r"\bleft (?:arm|hand|gripper)\b", text))
    right = bool(re.search(r"\bright (?:arm|hand|gripper)\b", text))
    if left == right:
        return None
    return "left" if left else "right"


def _trajectory_arm_mode(steps: Sequence[Mapping[str, Any]]) -> str:
    active: set[str] = set()
    for step in steps:
        if str(step["left_action"]) != "still":
            active.add("left")
        if str(step["right_action"]) != "still":
            active.add("right")
    if active == {"left"}:
        return "left"
    if active == {"right"}:
        return "right"
    if active == {"left", "right"}:
        return "bimanual"
    return "idle"


def _history_target(step: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "primary_action_verb": str(step["primary_action_verb"]),
        "sub_task": str(step["source_sub_task"]),
        "num_chunks": int(step["num_chunks"]),
    }


def candidate_prompt(
    *,
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    previous_step: Mapping[str, Any] | None,
    states: np.ndarray,
    revision_feedback: Sequence[str] = (),
) -> str:
    current = int(step["start_frame"])
    frozen_sub_task = str(
        step.get("source_sub_task")
        or _generic_subtask(str(step["left_action"]), str(step["right_action"]))
    )
    lines = [
        f"Task: {record['task_instruction']}",
        "Planning boundary: current recovery observation; no future episode length is supplied.",
        "Current 16-D end-effector state: "
        + compact_json(_named_state(states[current])),
        "Gripper convention: values near 1 mean open; values near 0 mean closed.",
        "Frozen motion primitive: "
        + compact_json(
            {
                "left_action": step["left_action"],
                "right_action": step["right_action"],
                "primary_action_verb": step["primary_action_verb"],
            }
        ),
        f"Frozen sub_task (copy this text exactly): {frozen_sub_task}",
        (
            "The sub_task field is fixed supervised metadata, not a free-form "
            "caption. Return it exactly as written above. Put any visually "
            "grounded object details or uncertainty only in reasoning, evidence, "
            "and uncertainties."
        ),
        (
            "The frozen primitive is supervised trajectory metadata. Use it to "
            "describe the concrete next atomic subtask, but justify the action "
            "from visible current/history evidence rather than merely restating "
            "the label. Do not predict or mention the segment endpoint, future "
            "images, future object effects, or final episode outcome."
        ),
        (
            "The frozen primitive is authoritative supervision for this segment; "
            "do not substitute an action that seems preferable. A numeric "
            "gripper value establishes open/closed configuration but does not by "
            "itself prove object contact, grasp success, release success, or "
            "support. When views are occluded, conflicting, or inconclusive, use "
            "conservative language such as attempt, move toward, remain "
            "stationary, or allow the object to settle, and record the unresolved "
            "physical state under uncertainties instead of asserting success."
        ),
    ]
    style_id = int(step.get("global_block_index", step["source_segment_id"])) % len(
        STYLE_SPECS
    )
    lines.extend(
        [
            f"Reasoning organization: {STYLE_SPECS[style_id]['name']}. {STYLE_SPECS[style_id]['brief']}",
            "Use a varied natural opening, not a repeated template. Do not invent a previous "
            "observation when history is absent. Explicitly distinguish both arms. A still "
            "arm is stationary, not necessarily holding or stabilizing an object. Avoid "
            "positive holding, secure-grasp and completed-success claims; describe visible "
            "configuration and uncertainty instead. Do not output a final Therefore action "
            "sentence: the program appends the exact action and chunk budget afterward.",
        ]
    )
    trajectory_mode = _trajectory_arm_mode(record["source_steps"])
    instruction_arm = _instruction_arm(str(record["task_instruction"]))
    arm_override = (
        trajectory_mode in {"left", "right"}
        and instruction_arm in {"left", "right"}
        and trajectory_mode != instruction_arm
    )
    if instruction_arm and trajectory_mode != "bimanual" and not arm_override:
        lines.append(f"Required task arm: {instruction_arm}")
    if trajectory_mode == "bimanual":
        lines.append(
            "Task arm mode: multi-arm recovery; either or both arms may act or "
            "hand off according to the frozen primitive. Any arm named in the "
            "original instruction applies to its relevant task phase and is not "
            "a whole-episode prohibition on the other arm."
        )
    if arm_override:
        lines.append(
            f"Original instruction arm mention: {instruction_arm}. "
            "Audited recovery-arm override: this recovery segment legitimately "
            f"uses the {trajectory_mode} arm to repair or continue from the "
            "failed state. Follow the frozen recovery primitive for this "
            "segment rather than rejecting it because the original instruction "
            "mentions the other arm."
        )
    if previous_step is None:
        lines.append(
            "History boundary: none. This is the first recovery planning "
            "boundary; infer the unresolved task condition only from the current "
            "images and state."
        )
    else:
        previous = int(previous_step["start_frame"])
        lines.extend(
            [
                "Previous command: " + compact_json(_history_target(previous_step)),
                "Previous boundary state: "
                + compact_json(_named_state(states[previous])),
                "Observed state delta: "
                + compact_json(_round_vector(states[current] - states[previous])),
                (
                    "Compare the previous command with the observed visual/state "
                    "change. Treat it as completed only when the evidence supports "
                    "advancing; otherwise describe the appropriate retry or "
                    "correction. The observed delta belongs to the already-issued "
                    "previous command; it is not the motion direction of the new "
                    "frozen primitive."
                ),
            ]
        )
    if revision_feedback:
        lines.append(
            "Critic revision requirements: " + compact_json(list(revision_feedback))
        )
    lines.append(
        "Return exactly one JSON object with keys: reasoning (80-180 words of "
        "causal pre-action reasoning), sub_task (one concrete sentence matching "
        "the frozen left/right primitive and visible task object), evidence "
        "(2-5 concise visible/state facts), and uncertainties (0-3 concise "
        "items). No markdown and no extra keys."
    )
    return "\n".join(lines)


def critic_prompt(
    *,
    record: Mapping[str, Any],
    step: Mapping[str, Any],
    previous_step: Mapping[str, Any] | None,
    states: np.ndarray,
    candidate: Mapping[str, Any],
    critic_index: int,
) -> str:
    current = int(step["start_frame"])
    frozen_sub_task = str(
        step.get("source_sub_task")
        or _generic_subtask(str(step["left_action"]), str(step["right_action"]))
    )
    context = [
        "Current 16-D end-effector state: "
        + compact_json(_named_state(states[current])),
        "Gripper convention: values near 1 mean open; values near 0 mean closed.",
    ]
    if previous_step is not None:
        previous = int(previous_step["start_frame"])
        context.extend(
            [
                "Previous command: " + compact_json(_history_target(previous_step)),
                "Previous boundary state: "
                + compact_json(_named_state(states[previous])),
                "Observed state delta: "
                + compact_json(_round_vector(states[current] - states[previous])),
            ]
        )
    return "\n".join(
        [
            f"Critic copy: {critic_index}",
            f"Task: {record['task_instruction']}",
            *context,
            "Frozen motion primitive: "
            + compact_json(
                {
                    "left_action": step["left_action"],
                    "right_action": step["right_action"],
                    "primary_action_verb": step["primary_action_verb"],
                }
            ),
            "Candidate: " + compact_json(candidate),
            f"Required frozen sub_task: {frozen_sub_task}",
            (
                "The sub_task is a canonical supervised action label. If it "
                "exactly matches the required frozen sub_task, treat "
                "primitive_match and concrete_sub_task as satisfied. Do not "
                "request object-name, grasp-state, or directional rewrites of "
                "that field; assess such visual grounding only in reasoning, "
                "evidence, and uncertainties."
            ),
            (
                "Audit the candidate against the supplied pre-action images. "
                "The final Therefore sentence is appended by the program from "
                "the frozen action and chunk budget; audit its action consistency, "
                "not its numeric budget as a visual claim. "
                "The candidate may use current images, current state, and prior "
                "boundary history only. It must not claim the frozen action has "
                "already happened, reveal a future endpoint/outcome, invent "
                "contact or success, or conflict with the active arm and motion "
                "primitive. The frozen primitive is authoritative; do not "
                "replace it with a different action based on your own plan. "
                "When history is present, its state delta describes the already-"
                "issued previous command, not the new frozen primitive. The "
                "sub_task must be concrete enough for planner "
                "training and must not use internal labels such as move_group."
            ),
            (
                "Treat the frozen primitive as authoritative trajectory "
                "supervision. Do not return REVISE merely because another action "
                "would be your preferred plan. Numeric gripper state proves only "
                "open/closed configuration, not object contact or successful "
                "grasp/release. If visual views are occluded, conflicting, or "
                "ambiguous, accept conservative wording that describes an "
                "attempt, motion toward a target, stationary pose, or settling "
                "without claiming an unobserved physical effect. Every REVISE "
                "verdict must include concrete revision_instructions; issues may "
                "not be left as the only explanation."
            ),
            (
                "Return exactly one JSON object with keys verdict (PASS or "
                "REVISE), checks (an object containing primitive_match, "
                "current_observation_grounded, history_grounded, "
                "no_future_leakage, no_unsupported_effect, concrete_sub_task, "
                "reasoning_supports_action; every value boolean), issues (list "
                "of strings), and revision_instructions (list of strings)."
            ),
        ]
    )


def _validate_candidate(
    value: Mapping[str, Any],
    *,
    step: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != {"reasoning", "sub_task", "evidence", "uncertainties"}:
        raise ValueError("candidate must use exactly four requested keys")
    reasoning = re.sub(r"\s+", " ", str(value["reasoning"])).strip()
    generated_sub_task = re.sub(r"\s+", " ", str(value["sub_task"])).strip()
    # ``sub_task`` is frozen trajectory supervision.  Model-generated wording
    # is deliberately not allowed to alter the training target; only the
    # reasoning/evidence fields are authored by the model.
    sub_task = re.sub(
        r"\s+",
        " ",
        str(step.get("source_sub_task") or generated_sub_task),
    ).strip()
    evidence = (
        [
            re.sub(r"\s+", " ", str(item)).strip()
            for item in value["evidence"]
            if str(item).strip()
        ]
        if isinstance(value["evidence"], list)
        else []
    )
    uncertainties = (
        [
            re.sub(r"\s+", " ", str(item)).strip()
            for item in value["uncertainties"]
            if str(item).strip()
        ]
        if isinstance(value["uncertainties"], list)
        else []
    )
    words = len(re.findall(r"\b[\w'-]+\b", reasoning))
    if not 55 <= words <= 230:
        raise ValueError(f"candidate reasoning has {words} words")
    if not generated_sub_task or len(generated_sub_task) > 300:
        raise ValueError("candidate sub_task is empty or too long")
    if not 1 <= len(evidence) <= 6:
        raise ValueError("candidate evidence must contain 1-6 items")
    if len(uncertainties) > 4:
        raise ValueError("candidate uncertainties must contain at most 4 items")
    joined = " ".join((reasoning, sub_task, *evidence, *uncertainties)).lower()
    forbidden = (
        "future image",
        "end frame",
        "segment endpoint",
        "ground truth",
        "move_group",
        "semantic_action",
        "guide_action",
    )
    if any(token in joined for token in forbidden):
        raise ValueError("candidate contains hidden/future supervision language")
    if "global_block_index" in step:
        if generated_sub_task != sub_task:
            raise ValueError(
                "candidate must copy the canonical frozen sub_task exactly"
            )
        validate_body(
            reasoning, step, int(step["global_block_index"]) % len(STYLE_SPECS)
        )
        reasoning = complete_reasoning(reasoning, step)
    required_terms = {
        "close_gripper": ("close", "grasp", "secure", "grip", "clamp"),
        "open_gripper": ("open", "release", "unclamp", "spread"),
        "lift": ("lift", "raise", "upward"),
        "lower": ("lower", "descend", "downward"),
        "turn": ("turn", "rotate", "reorient", "twist"),
        "move": (
            "move",
            "approach",
            "reposition",
            "align",
            "translate",
            "shift",
            "advance",
            "carry",
            "bring",
            "position",
            "transfer",
            "transport",
            "withdraw",
            "retreat",
            "clear",
        ),
        "still": (
            "still",
            "hold",
            "wait",
            "observe",
            "verify",
            "maintain",
            "stationary",
            "remain",
            "stabilize",
            "pause",
        ),
    }
    for arm in ("left", "right"):
        action = str(step[f"{arm}_action"])
        other = str(step[("right" if arm == "left" else "left") + "_action"])
        if action == "still" and other != "still":
            continue
        if action != "still" and arm not in sub_task.lower():
            raise ValueError(f"sub_task does not identify active {arm} arm")
        if not any(term in sub_task.lower() for term in required_terms[action]):
            raise ValueError(f"sub_task does not express {arm}:{action}")
    return {
        "segment_id": int(step["source_segment_id"]),
        "reasoning": reasoning,
        "sub_task": sub_task,
        "action_description": sub_task,
        "evidence": evidence,
        "uncertainties": uncertainties,
    }


def _validate_critic(value: Mapping[str, Any]) -> dict[str, Any]:
    verdict = str(value.get("verdict", "")).upper()
    checks = value.get("checks")
    issues = value.get("issues")
    instructions = value.get("revision_instructions")
    expected = {
        "primitive_match",
        "current_observation_grounded",
        "history_grounded",
        "no_future_leakage",
        "no_unsupported_effect",
        "concrete_sub_task",
        "reasoning_supports_action",
    }
    if verdict not in {"PASS", "REVISE"}:
        raise ValueError("critic verdict must be PASS or REVISE")
    if not isinstance(checks, Mapping) or set(checks) != expected:
        raise ValueError("critic checks mismatch")
    if any(not isinstance(checks[key], bool) for key in expected):
        raise ValueError("critic check values must be booleans")
    normalized_checks = {key: checks[key] for key in sorted(expected)}
    normalized_issues = [
        str(item).strip() for item in issues or [] if str(item).strip()
    ]
    normalized_instructions = [
        str(item).strip() for item in instructions or [] if str(item).strip()
    ]
    if verdict == "PASS" and (not all(normalized_checks.values()) or normalized_issues):
        raise ValueError("PASS critic contains failed checks or issues")
    if verdict == "REVISE" and not normalized_instructions:
        # Some otherwise useful critic responses put the requested edits in
        # ``issues`` but leave ``revision_instructions`` empty. Preserve those
        # edits instead of discarding the whole episode. If the critic only
        # supplied failed boolean checks, turn them into explicit feedback so
        # the candidate still receives a concrete, auditable revision target.
        normalized_instructions = list(normalized_issues)
        if not normalized_instructions:
            normalized_instructions = [
                "Revise the candidate to satisfy the failed critic check: "
                + key.replace("_", " ")
                + "."
                for key, passed in normalized_checks.items()
                if not passed
            ]
        if not normalized_instructions:
            raise ValueError(
                "REVISE critic requires issues, failed checks, or revision instructions"
            )
    return {
        "verdict": verdict,
        "checks": normalized_checks,
        "issues": normalized_issues,
        "revision_instructions": normalized_instructions,
    }


def _template_candidate(
    record: Mapping[str, Any], step: Mapping[str, Any]
) -> dict[str, Any]:
    primitive = str(step["primary_action_verb"])
    sub_task = _generic_subtask(str(step["left_action"]), str(step["right_action"]))
    reasoning = (
        f"The current recovery observation must be evaluated against the task, "
        f"{str(record['task_instruction']).rstrip('.')}. The visible robot pose "
        "and gripper configuration define a fresh planning boundary, while no "
        "future state or outcome is available. The next supervised primitive is "
        f"{primitive}, so the robot should execute one bounded atomic correction "
        "that preserves the inactive arm and advances the unresolved object "
        "interaction. The command should be checked at the next observation "
        "before any later step assumes contact, grasp security, placement, or "
        "task completion."
    )
    return {
        "segment_id": int(step["source_segment_id"]),
        "reasoning": reasoning,
        "sub_task": sub_task,
        "action_description": sub_task,
        "evidence": [
            "Current multiview observation is available",
            "Current 16-D robot state is available",
        ],
        "uncertainties": ["The physical effect of the next action is not yet observed"],
    }
