"""Recovery-specific language and release gates shared by all native stages.

Object names and evidence-backed failure descriptions are allowed. The normal
pipeline's object/color blacklist is deliberately not imposed on recovery.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Mapping

from atomic_seg.action_language import cot_final_action_sentence
from atomic_seg.common.cot_format import word_count
from atomic_seg.cot_data.styles import STYLE_SPECS
from atomic_seg.cot_data.validation import _contradictory_action_issues

from .policy import QUALITY_POLICY

FINAL_PREFIX = "Therefore, the next action should be:"


def adapter_final_sentence(target: Mapping[str, Any] | None) -> str:
    """Exact-label conclusion for arbitrary-layout experimental adapters."""
    if target is None:
        return "Therefore, no next action is supervised in this diagnostic sample."
    import json

    chunks = target["num_chunks"]
    if type(chunks) is not int or chunks < 1:
        raise ValueError("adapter action chunk budget must be a positive integer")
    return (
        FINAL_PREFIX
        + " "
        + json.dumps(str(target["sub_task"]), ensure_ascii=False)
        + f" ({target['primary_action_verb']}), executed for {chunks} action chunk(s)."
    )


def actions(target: Mapping[str, Any]) -> dict[str, str]:
    parts = [
        item.strip().split(":")
        for item in str(target["primary_action_verb"]).split(",")
    ]
    if len(parts) != 2 or any(len(part) != 2 for part in parts):
        raise ValueError("invalid recovery action pair")
    result = dict(parts)
    if set(result) != {"left", "right"}:
        raise ValueError("recovery target requires both arms")
    return result


def final_sentence(target: Mapping[str, Any]) -> str:
    pair = actions(target)
    return cot_final_action_sentence(
        pair["left"], pair["right"], int(target["num_chunks"])
    )


def reasoning_body(
    reasoning: str, target: Mapping[str, Any], *, required: bool = True
) -> str:
    value = re.sub(r"\s+", " ", reasoning).strip()
    closing = final_sentence(target)
    if value.endswith(closing) and value.count(FINAL_PREFIX) == 1:
        return value[: -len(closing)].strip()
    if required or FINAL_PREFIX.lower() in value.lower():
        raise ValueError("CoT final action sentence does not match target")
    return value


def complete_reasoning(body: str, target: Mapping[str, Any]) -> str:
    return reasoning_body(body, target, required=False) + " " + final_sentence(target)


def validate_body(
    body: str, target: Mapping[str, Any], style_id: int | None = None
) -> None:
    if not 55 <= word_count(body) <= 230:
        raise ValueError("recovery reasoning body must have 55-230 words")
    forbidden = (
        r"<think>|</think>|```",
        r"\b(?:guide_action|ground truth|move_group|semantic_action)\b",
        r"\b(?:future (?:frame|image|state)|segment endpoint|frozen (?:target|primitive)|supervised primitive|source reasoning|style brief)\b",
        r"therefore, the next action should be",
    )
    if any(re.search(pattern, body, re.I) for pattern in forbidden):
        raise ValueError("recovery CoT exposes supervision or nested output formatting")
    # Still does not mean holding an object; closed grippers do not establish a
    # grasp. Prefer uncertainty over positive grasp/success assertions. Retain
    # failure descriptions and ordinary visible object facts for recovery.
    for sentence in re.split(r"(?<=[.!?])\s+", body):
        clauses = re.split(r"\b(?:but|however|yet|although)\b|;", sentence, flags=re.I)
        for clause in clauses:
            positive = re.search(
                r"\b(?:holds?|holding|carrying|grasping|stabilizes?|stabilizing)\s+(?:a|an|the|this|that)\b|"
                r"\b(?:has|have|had)\s+(?:successfully\s+)?(?:grasped|secured|released|placed)\b|"
                r"\b(?:secure|stable|firm) grasp\b|\bcompleted successfully\b|"
                r"\b(?:task|placement|release|grasp)\s+(?:is|was|has been)\s+(?:complete|confirmed|successful)\b",
                clause,
                re.I,
            )
            negated = re.search(
                r"\b(?:cannot|can't|does not|do not|did not|not|without|uncertain|unconfirmed|unverified)\b",
                clause[: positive.start()] if positive else "",
                re.I,
            )
            if positive and not negated:
                raise ValueError(
                    "unverified holding/grasp/success assertion in recovery CoT"
                )
    issues = _contradictory_action_issues(body, target)
    # Gripper commands are often phrased "right gripper should close" or
    # "should close the right gripper", unlike Cartesian arm commands.
    pair = actions(target)
    for arm, expected in pair.items():
        for verb, action in (("open", "open_gripper"), ("close", "close_gripper")):
            pattern = (
                rf"\b{arm} gripper (?:must|should|will|needs? to) {verb}\b|"
                rf"\b(?:must|should|will|needs? to) {verb} the {arm} gripper\b"
            )
            if re.search(pattern, body, re.I) and expected != action:
                issues.append(f"The {arm} gripper command contradicts {expected}.")
    if issues:
        raise ValueError("; ".join(issues))
    if any(not re.search(rf"\b{arm}\b", body, re.I) for arm in pair):
        raise ValueError("recovery CoT must distinguish both arms")
    if style_id is not None:
        # Same five conceptual styles as normal CoT; no fixed sentence template.
        from .styles import _target_semantic_issues

        issues = _target_semantic_issues(body, target, style_id)
        if issues:
            raise ValueError("; ".join(issues))


def annotation_quality(
    candidate: Mapping[str, Any], step: Mapping[str, Any]
) -> dict[str, Any]:
    style_id = int(step["global_block_index"]) % len(STYLE_SPECS)
    body = reasoning_body(str(candidate["reasoning"]), step)
    validate_body(body, step, style_id)
    return {
        "quality_policy": QUALITY_POLICY,
        "style_id": style_id,
        "style_name": STYLE_SPECS[style_id]["name"],
        "reasoning_sha256": hashlib.sha256(
            str(candidate["reasoning"]).encode()
        ).hexdigest(),
        "body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }


class DiversityAudit:
    def __init__(self) -> None:
        self.styles: Counter[int] = Counter()
        self.hashes: set[str] = set()
        self.openings: Counter[str] = Counter()

    def add(self, reasoning: str, target: Mapping[str, Any], block_index: int) -> None:
        body = reasoning_body(reasoning, target)
        style_id = block_index % len(STYLE_SPECS)
        validate_body(body, target, style_id)
        digest = hashlib.sha256(body.lower().encode()).hexdigest()
        if digest in self.hashes:
            raise ValueError(
                "duplicate recovery CoT body (ignoring final action sentence)"
            )
        self.hashes.add(digest)
        self.styles[style_id] += 1
        words = re.findall(r"[a-z0-9']+", body.lower())
        self.openings[" ".join(words[:8])] += 1

    def report(self) -> dict[str, Any]:
        count = sum(self.styles.values())
        if not count:
            raise ValueError("empty recovery CoT release")
        expected = {i: count // 5 + (i < count % 5) for i in range(5)}
        if any(self.styles[i] != expected[i] for i in expected):
            raise ValueError("recovery five-style allocation mismatch")
        if len(self.openings) < min(count, max(10000, count // 10)):
            raise ValueError("recovery opening diversity is too low")
        if max(self.openings.values()) > math.ceil(count * 0.02):
            raise ValueError("one recovery opening is over-represented")
        return {
            "passed": True,
            "quality_policy": QUALITY_POLICY,
            "rows": count,
            "styles": {STYLE_SPECS[i]["name"]: self.styles[i] for i in range(5)},
            "unique_reasoning_bodies": len(self.hashes),
            "opening_unique": len(self.openings),
            "opening_max_count": max(self.openings.values()),
        }
