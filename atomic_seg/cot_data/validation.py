"""Normal CoT envelope and explicit action consistency checks.

Visual objects, colors, contact and ownership are checked by the grounded
critics. They are not vocabulary errors.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from atomic_seg.common.cot_format import word_count as _word_count

from .styles import STYLE_SPECS

FORBIDDEN_REASONING_PATTERNS = (
    r"\bguide_action\b",
    r"\bground truth\b",
    r"\bfuture (?:frame|image|state|endpoint)\b",
    r"\bsegment endpoint\b",
    r"\bfrozen target\b",
    r"\bsource reasoning\b",
    r"\bstyle brief\b",
    r"<think>|</think>",
    r"therefore, the next action should be",
    r"\b(?:authoritative|supervision|supervisory)\b",
    r"\bfixed (?:command|action|target)\b",
    r"\b(?:task arm constraint|single-arm mode|multi-arm mode|single-arm constraint|multi-arm constraint)\b",
    r"\b(?:task|arm) constraints?\b",
    r"\b(?:left|right) arm (?:is )?(?:designated|constrained)\b",
    r"\bsole active manipulator\b",
    r"\bcontrol system dictates\b",
    r"\btherefore[:,.]?\s*(?:left:[a-z_]+,\s*right:[a-z_]+)?[.!]?\s*$",
    r"\bleft:[a-z_]+,\s*right:[a-z_]+\b",
    r"\b(?:state vector|state channels|rotation components)\b",
    r"\b(?:control logic|control sequence|command specifies|plan requires)\b",
    r"\bTEACHER_ONLY\b",
    r"\b(?:annotation|annotations|dataset provenance|evidence_ids|claim_evidence)\b",
    r"\b(?:observer card|critic verdict|style label)\b",
)


def _parse_reasoning(value: Mapping[str, Any]) -> str:
    if not isinstance(value, Mapping) or set(value) != {"reasoning"}:
        raise ValueError("response must contain exactly one reasoning field")
    if not isinstance(value["reasoning"], str):
        raise ValueError("reasoning must be a string")
    reasoning = value["reasoning"].strip()
    count = _word_count(reasoning)
    if not 55 <= count <= 150:
        raise ValueError(f"reasoning has {count} words; expected 55-150")
    for pattern in FORBIDDEN_REASONING_PATTERNS:
        if re.search(pattern, reasoning, re.IGNORECASE):
            raise ValueError(f"reasoning contains forbidden claim: {pattern}")
    if reasoning.startswith(("Evidence:", "Analysis:", "Reasoning:", "Style:")):
        raise ValueError("reasoning must be natural prose without a heading")
    return reasoning


def _actions(target: Mapping[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in str(target["primary_action_verb"]).split(","):
        arm, separator, action = item.strip().partition(":")
        if not separator or arm not in {"left", "right"} or arm in output:
            raise ValueError("invalid primary_action_verb")
        output[arm] = action
    if set(output) != {"left", "right"}:
        raise ValueError("primary_action_verb must contain both arms")
    return output


_INTENDED_ACTION_TERMS = {
    "move": r"(?:move|reposition|translate|shift|travel|advance)",
    "turn": r"(?:turn|rotate|reorient)",
    "lift": r"(?:lift|raise|ascend|move upward|travel upward)",
    "lower": r"(?:lower|descend|move downward|travel downward)",
    "open_gripper": (
        r"(?:open (?:its |the )?gripper|actuate (?:its |the )?gripper "
        r"(?:to|into) (?:an? )?open(?: configuration| state)?)"
    ),
    "close_gripper": (
        r"(?:close (?:its |the )?gripper|actuate (?:its |the )?gripper "
        r"(?:to|into) (?:a )?closed(?: configuration| state)?)"
    ),
    "oscillate": r"(?:oscillate|shake|perform (?:an )?alternating motion)",
    "still": (
        r"(?:(?:remain|stay|be) (?:completely )?"
        r"(?:still|stationary|idle|inactive|fixed)|"
        r"keep (?:its |the )?(?:pose|position)(?: unchanged)?)"
    ),
}
_INTENT_MODAL = (
    r"(?:must|should|will|needs? to|has to|is required to|is expected to|"
    r"is scheduled to|is planned to|is going to)(?: now)?"
)
_INTENT_LEAD = (
    r"(?:now )?(?:(?:execute|perform|make|initiate|begin|continue|undergo|"
    r"carry out) )?(?:(?:a|an|the) )?"
)


def _explicit_intended_actions(reasoning: str, arm: str) -> set[str]:
    """Return explicit future commands assigned to one arm in the prose."""

    subject = rf"(?:the )?{arm} (?:arm|gripper|end[- ]effector)(?!['’]s)"
    between = r"(?:\s+(?:now|currently|therefore|consequently))?\s+"
    intended: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+", reasoning):
        lower_sentence = sentence.lower()
        for action, terms in _INTENDED_ACTION_TERMS.items():
            subject_first = re.compile(
                rf"\b{subject}\b{between}{_INTENT_MODAL}\s+{_INTENT_LEAD}"
                rf"{terms}\b",
                re.IGNORECASE,
            )
            action_first = re.compile(
                rf"\b{_INTENT_MODAL}\s+{_INTENT_LEAD}{terms}\s+"
                rf"(?:the )?{arm} (?:arm|gripper|end[- ]effector)\b",
                re.IGNORECASE,
            )
            matches = [
                *subject_first.finditer(sentence),
                *action_first.finditer(sentence),
            ]
            for match in matches:
                prefix = lower_sentence[: match.start()]
                hypothetical = re.search(
                    r"\b(?:whether|if|might|could|would|may)\b[^,.!?;:]*$",
                    prefix,
                )
                negated = re.search(
                    r"\b(?:not|no)\b[^,.!?;:]{0,16}$",
                    prefix,
                )
                rejected_alternative = re.search(
                    r"\b(?:alternative|option|hypothesis|might|could|would)\b",
                    prefix,
                ) and re.search(
                    r"\b(?:reject(?:ed)?|ruled out|unnecessary|invalid|"
                    r"no evidence|does not need|do not need|not needed)\b",
                    lower_sentence[match.end() :],
                )
                if not rejected_alternative and not hypothetical and not negated:
                    intended.add(action)
                    break
    return intended


def _contradictory_action_issues(
    reasoning: str, target: Mapping[str, Any]
) -> list[str]:
    issues: list[str] = []
    opposites = {
        "lift": {"lower"},
        "lower": {"lift"},
        "open_gripper": {"close_gripper"},
        "close_gripper": {"open_gripper"},
    }
    for arm, expected in _actions(target).items():
        intended = _explicit_intended_actions(reasoning, arm)
        if expected == "still":
            conflicting = intended - {"still"}
        elif expected in {"move", "lift", "lower", "oscillate"}:
            conflicting = (intended & opposites.get(expected, set())) | (
                intended & {"still"}
            )
        elif expected == "turn":
            # A turn can intentionally keep Cartesian position fixed.
            conflicting = intended & opposites.get(expected, set())
        else:
            # Gripper actions can correctly leave the Cartesian arm pose fixed.
            conflicting = intended & opposites.get(expected, set())
        if conflicting:
            issues.append(
                f"The {arm} arm is labeled {expected}, but the reasoning "
                f"explicitly commands {', '.join(sorted(conflicting))}."
            )
    return issues


def _semantic_issues(
    reasoning: str,
    target: Mapping[str, Any],
    style_id: int,
) -> list[str]:
    issues: list[str] = []
    issues.extend(_contradictory_action_issues(reasoning, target))
    if not 0 <= style_id < len(STYLE_SPECS):
        issues.append("Unknown reasoning style.")
    return issues
