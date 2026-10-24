"""Freeze student evidence and validate the four annotation stages."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from atomic_seg.action_language import cot_final_action_sentence
from atomic_seg.common.artifacts import compact_json, sha256_file
from atomic_seg.prompt_contract import THINKING_USER_ENDING, system_prompt_for_mode

from .prompts import PROMPT_VERSION, STYLE_HINTS
from .validation import _actions, _parse_reasoning, _semantic_issues, _word_count

QUALITY_POLICY = "robotwin_visual_grounded_dual_critic_v1"
ANNOTATION_SCHEMA = "robotwin_grounded_cot_annotation_v1"
FIELDS = {"guide_action", "primary_action_verb", "sub_task", "num_chunks"}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def freeze_row(row, *, check_images=True):
    messages = row.get("messages")
    if not isinstance(messages, list) or [m.get("role") for m in messages] != [
        "system", "user", "assistant",
    ]:
        raise ValueError("Require exactly system/user/assistant messages")
    if not all(isinstance(m.get("content"), str) for m in messages):
        raise ValueError("Message contents must be strings")
    mode = row.get("task_arm_mode")
    if messages[0]["content"] != system_prompt_for_mode(str(mode)):
        raise ValueError("Source system prompt is not canonical")
    answer = messages[-1]["content"]
    if answer.startswith("<think>"):
        if answer.count("<think>") != 1 or answer.count("</think>") != 1:
            raise ValueError("Source assistant has an invalid think envelope")
        final_text = answer.split("</think>", 1)[1].strip()
    else:
        final_text = answer.strip()
    target = json.loads(final_text)
    if not isinstance(target, dict) or set(target) != FIELDS:
        raise ValueError("Final JSON must contain exactly four action fields")
    guide = target["guide_action"]
    if not isinstance(guide, list) or len(guide) != 16 or any(
        type(v) not in (int, float) or not math.isfinite(v) for v in guide
    ):
        raise ValueError("guide_action must contain 16 finite components")
    if type(target["num_chunks"]) is not int or target["num_chunks"] < 1:
        raise ValueError("num_chunks must be a positive integer")
    if not isinstance(target["sub_task"], str) or not target["sub_task"].strip():
        raise ValueError("sub_task must be nonempty")
    actions = _actions(target)
    if any(a not in {"still", "move", "turn", "lift", "lower", "open_gripper",
                     "close_gripper", "oscillate"} for a in actions.values()):
        raise ValueError("Unsupported action primitive")
    if row.get("target") != target:
        raise ValueError("Source target differs from the final answer")
    paths = row.get("images")
    if not isinstance(paths, list) or len(paths) not in (3, 6) or not all(
        isinstance(p, str) for p in paths
    ):
        raise ValueError("Require three current or six history/current images")
    user = messages[1]["content"]
    if user.count("<image>") != len(paths) or not user.endswith(THINKING_USER_ENDING):
        raise ValueError("Image count or thinking user ending mismatch")
    if len(paths) == 6 and (
        "Previous boundary state: " not in user or "Previous command: " not in user
    ):
        raise ValueError("History images require visible boundary history")
    if len(paths) == 3 and "History boundary: none;" not in user:
        raise ValueError("Initial images require an explicit initial boundary")
    if not any(line.startswith("Current state: ") for line in user.splitlines()):
        raise ValueError("Current state missing")
    cameras = ("head", "left_wrist", "right_wrist")
    roles = [f"current.{c}" for c in cameras]
    if len(paths) == 6:
        roles = [f"history.{c}" for c in cameras] + roles
    images = []
    for role, name in zip(roles, paths):
        path = Path(name)
        if check_images and (
            not path.is_absolute() or not path.is_file() or not path.stat().st_size
        ):
            raise ValueError(f"Image missing or not absolute: {path}")
        if check_images:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
        images.append({"id": role, "path": name,
                       "sha256": sha256_file(path) if check_images else None})
    visible = {
        "student_system": messages[0]["content"], "student_user": user,
        "images": images,
        "text_evidence_ids": [f"user.line.{i + 1}" for i in range(len(user.splitlines()))],
    }
    frozen = {
        "source_row_sha256": digest(row), "visible": visible,
        "target": copy.deepcopy(target), "final_json_text": final_text,
        "final_json_sha256": hashlib.sha256(final_text.encode()).hexdigest(),
        "student_visible_sha256": digest(visible),
        "closing_sentence": cot_final_action_sentence(
            actions["left"], actions["right"], target["num_chunks"]
        ),
    }
    visible_payload(frozen)
    return frozen


def visible_payload(frozen):
    visible = frozen["visible"]
    user = visible["student_user"]
    aliases, facts = {}, {}
    for index, line in enumerate(user.splitlines(), 1):
        for prefix, key in (
            ("Current state: ", "current.state"),
            ("Previous boundary state: ", "previous.state"),
        ):
            if not line.startswith(prefix):
                continue
            if key in aliases:
                raise ValueError("Duplicate visible state field")
            state = json.loads(line[len(prefix):])
            if set(state) != {"left", "right"}:
                raise ValueError("Visible state must name both arms")
            aliases[key] = f"user.line.{index}"
            facts[key] = {}
            for arm, values in state.items():
                aperture = values["gripper"]
                if type(aperture) not in (int, float) or not math.isfinite(aperture):
                    raise ValueError("Invalid visible aperture")
                label = "closed" if aperture <= 0.05 else "open" if aperture >= 0.95 else "partially_open"
                facts[key][arm] = {"aperture_value": aperture, "aperture_label": label}
    return {
        "student_system": visible["student_system"], "student_user": user,
        "evidence_aliases": aliases, "INPUT_DERIVED_ROBOT_FACTS": facts,
        "image_roles": [
            {"id": image["id"], "temporal_role": image["id"].split(".")[0],
             "camera": image["id"].split(".")[1]} for image in visible["images"]
        ],
        "user_text_lines": [
            {"id": f"user.line.{i + 1}", "text": text}
            for i, text in enumerate(user.splitlines())
        ],
    }


def request_payload(stage, frozen, *, observer=None, reasoning=None,
                    style="evidence_first", feedback=None, previous_draft=None):
    if stage not in {"observe", "write", "grounding_critic", "action_critic"}:
        raise ValueError("Unknown annotation stage")
    payload = {"student_visible": visible_payload(frozen)}
    if stage in {"write", "action_critic"}:
        payload["TEACHER_ONLY_PLANNED_ACTION"] = {
            k: frozen["target"][k] for k in ("primary_action_verb", "sub_task", "num_chunks")
        }
    if stage == "write":
        payload.update(observer_evidence=observer, style_hint=STYLE_HINTS[style])
        if previous_draft is not None:
            payload["last_candidate_to_revise"] = previous_draft
    if stage in {"grounding_critic", "action_critic"}:
        payload["candidate_reasoning"] = reasoning
    if feedback:
        payload["revision_feedback"] = feedback
    return payload


def evidence_refs(frozen):
    return (
        {image["id"] for image in frozen["visible"]["images"]}
        | set(frozen["visible"]["text_evidence_ids"])
        | set(visible_payload(frozen)["evidence_aliases"])
    )


def obvious_aperture_conflict(text, frozen):
    state = visible_payload(frozen)["INPUT_DERIVED_ROBOT_FACTS"].get("current.state", {})
    for arm, fact in state.items():
        wrong = {"closed": "open", "open": "closed"}.get(fact["aperture_label"])
        if wrong is None:
            continue
        pattern = rf"\b{arm} gripper (?:is|reads|is currently|is now|remains) (?:fully |currently )?{wrong}\b"
        if re.search(pattern, text, re.I):
            raise ValueError(f"Current {arm} gripper is {fact['aperture_label']}, not {wrong}")


def refs(item, allowed, *, required=False):
    values = item.get("evidence_ids")
    if not isinstance(values, list) or not all(
        isinstance(v, str) and v in allowed for v in values
    ) or (required and not values):
        raise ValueError("Unknown, malformed or missing evidence_ids")
    return values


def normalize_plan_reference(value, stage):
    result, changes = copy.deepcopy(value), []
    if not isinstance(result, dict):
        return result, changes
    entries = []
    if stage == "write":
        entries.extend((c, "claim_evidence") for c in result.get("claim_evidence", [])
                       if isinstance(c, dict) and c.get("kind") == "planned_intent")
    if stage in {"write", "action_critic"}:
        entries.extend((c, "issues") for c in result.get("issues", []) if isinstance(c, dict))
    for item, field in entries:
        old = item.get("evidence_ids")
        if not isinstance(old, list):
            continue
        new = ["TEACHER_ONLY_PLANNED_ACTION" if v == "teacher.planned_action" else v for v in old]
        if new != old:
            changes.append({"field": field + ".evidence_ids", "before": old, "after": new})
            item["evidence_ids"] = new
    return result, changes


def validate_observer(value, frozen):
    if not isinstance(value, dict) or set(value) != {"task_goal", "observations", "unknowns"}:
        raise ValueError("Observer schema")
    if not isinstance(value["task_goal"], str) or not value["task_goal"].strip():
        raise ValueError("Observer missing task goal")
    if not isinstance(value["observations"], list) or not value["observations"] or not isinstance(value["unknowns"], list):
        raise ValueError("Observer evidence lists")
    ids, allowed = set(), evidence_refs(frozen)
    image_ids = {image["id"] for image in frozen["visible"]["images"]}
    for item in value["observations"]:
        if not isinstance(item, dict) or set(item) != {"id", "claim", "kind", "evidence_ids"}:
            raise ValueError("Observation schema")
        if not isinstance(item["id"], str) or not re.fullmatch(r"O[1-9]\d*", item["id"]) or item["id"] in ids:
            raise ValueError("Observation id malformed or duplicated")
        ids.add(item["id"])
        if not isinstance(item["claim"], str) or not item["claim"].strip():
            raise ValueError("Observation claim missing")
        if item["kind"] not in {"visual", "robot_state", "observed_change", "task_requirement"}:
            raise ValueError("Unknown observation kind")
        cited = refs(item, allowed, required=True)
        if item["kind"] == "visual" and not image_ids.intersection(cited):
            raise ValueError("Visual observation must cite an image")
        if item["kind"] == "robot_state":
            obvious_aperture_conflict(item["claim"], frozen)
    for item in value["unknowns"]:
        if not isinstance(item, dict) or set(item) != {"claim", "reason", "evidence_ids"}:
            raise ValueError("Unknown evidence schema")
        if not all(isinstance(item[k], str) and item[k].strip() for k in ("claim", "reason")):
            raise ValueError("Unknown claim or reason missing")
        refs(item, allowed, required=True)


def validate_writer(value, frozen, observer):
    if not isinstance(value, dict) or set(value) != {"status", "reasoning", "claim_evidence", "issues"}:
        raise ValueError("Writer schema")
    if value["status"] not in {"CANDIDATE", "TARGET_CONFLICT", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError("Writer status")
    if not isinstance(value["reasoning"], str) or not isinstance(value["claim_evidence"], list) or not isinstance(value["issues"], list):
        raise ValueError("Writer types")
    allowed = evidence_refs(frozen) | {o["id"] for o in observer["observations"]}
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"category", "explanation", "evidence_ids"}:
            raise ValueError("Writer issue schema")
        refs(issue, allowed | {"TEACHER_ONLY_PLANNED_ACTION"}, required=True)
    if value["status"] != "CANDIDATE":
        if value["reasoning"] or not value["issues"]:
            raise ValueError("Conflict requires empty prose and an evidence-linked issue")
        return
    if value["issues"]:
        raise ValueError("Candidate has unresolved writer issues")
    body = _parse_reasoning({"reasoning": value["reasoning"]})
    if body != value["reasoning"]:
        raise ValueError("Reasoning must have no surrounding whitespace")
    obvious_aperture_conflict(body, frozen)
    if not value["claim_evidence"]:
        raise ValueError("Writer must supply claim evidence")
    for claim in value["claim_evidence"]:
        if not isinstance(claim, dict) or set(claim) != {"quote", "kind", "evidence_ids"}:
            raise ValueError("Writer claim schema")
        if not isinstance(claim["quote"], str) or not claim["quote"] or claim["quote"] not in body:
            raise ValueError("Writer quote is not an exact body substring")
        if claim["kind"] not in {"observation", "task_requirement", "inference", "uncertainty", "planned_intent"}:
            raise ValueError("Writer claim kind")
        refs(claim, allowed | (
            {"TEACHER_ONLY_PLANNED_ACTION"} if claim["kind"] == "planned_intent" else set()
        ), required=True)


def validate_critic(value, frozen, body, stage):
    expected = {"verdict", "issues", "grounded_claims"} if stage == "grounding_critic" else {"verdict", "issues"}
    if stage not in {"grounding_critic", "action_critic"} or not isinstance(value, dict) or set(value) != expected:
        raise ValueError("Critic schema")
    verdicts = {"PASS", "REVISE", "INSUFFICIENT_EVIDENCE"}
    if stage == "action_critic":
        verdicts.add("TARGET_CONFLICT")
    if value["verdict"] not in verdicts or not isinstance(value["issues"], list):
        raise ValueError("Critic verdict")
    if (value["verdict"] == "PASS") != (not value["issues"]):
        raise ValueError("Critic verdict/issues mismatch")
    allowed = evidence_refs(frozen) | (
        {"TEACHER_ONLY_PLANNED_ACTION"} if stage == "action_critic" else set()
    )
    for issue in value["issues"]:
        if not isinstance(issue, dict) or set(issue) != {"quote", "category", "reason", "evidence_ids"}:
            raise ValueError("Critic issue schema")
        if not isinstance(issue["quote"], str) or issue["quote"] not in body:
            raise ValueError("Critic quote is not an exact body substring")
        if not issue["quote"] and issue["category"] != "missing_task_rationale":
            raise ValueError("Empty critic quote requires missing_task_rationale")
        refs(issue, allowed, required=True)
    if stage == "grounding_critic":
        if not isinstance(value["grounded_claims"], list):
            raise ValueError("grounded_claims type")
        for claim in value["grounded_claims"]:
            if not isinstance(claim, dict) or set(claim) != {"quote", "evidence_ids"} or not isinstance(claim["quote"], str) or not claim["quote"] or claim["quote"] not in body:
                raise ValueError("Grounded claim quote mismatch")
            refs(claim, allowed, required=True)
        if value["verdict"] == "PASS" and not value["grounded_claims"]:
            raise ValueError("Grounding PASS requires supported claims")


def validate_annotation(annotation, row, *, frozen=None, verify_receipts=True,
                        journal_root=None):
    frozen = frozen or freeze_row(row)
    if annotation.get("schema_version") != ANNOTATION_SCHEMA or annotation.get("quality_policy") != QUALITY_POLICY or annotation.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Annotation lacks the current observer and dual critic contract")
    if annotation.get("fallback") or type(annotation.get("accepted_round")) is not int or annotation["accepted_round"] < 1:
        raise ValueError("Fallback or unaccepted annotation")
    for key in ("student_visible_sha256", "final_json_sha256"):
        if annotation.get("frozen_hashes", {}).get(key) != frozen[key]:
            raise ValueError(f"Annotation input changed: {key}")
    body = _parse_reasoning({"reasoning": annotation["reasoning"]})
    if annotation["word_count"] != _word_count(body) or annotation["reasoning_sha256"] != hashlib.sha256(body.encode()).hexdigest():
        raise ValueError("Reasoning count or hash mismatch")
    style_id = annotation["style_id"]
    if type(style_id) is not int or not 0 <= style_id < len(STYLE_HINTS) or annotation["style_name"] != list(STYLE_HINTS)[style_id]:
        raise ValueError("Annotation style mismatch")
    issues = _semantic_issues(body, frozen["target"], style_id)
    if issues:
        raise ValueError("CoT/action mismatch: " + compact_json(issues))
    observer = annotation["observer"]
    validate_observer(observer, frozen)
    accepted = [r for r in annotation["rounds"] if r["round"] == annotation["accepted_round"]]
    if len(accepted) != 1:
        raise ValueError("Accepted candidate round missing or duplicated")
    candidate = accepted[0]
    if candidate.get("status") != "PASS":
        raise ValueError("Accepted candidate has no PASS marker")
    validate_writer(candidate["writer"], frozen, observer)
    if candidate["writer"]["status"] != "CANDIDATE" or candidate["writer"]["reasoning"] != body:
        raise ValueError("Accepted writer differs from annotation")
    critics = candidate["critics"]
    if set(critics) != {"grounding_critic", "action_critic"}:
        raise ValueError("Both independent critics are required")
    stages = [("observe", observer, annotation["observer_receipt"]),
              ("write", candidate["writer"], candidate["writer_receipt"])]
    for stage, review in critics.items():
        validate_critic(review["review"], frozen, body, stage)
        if review["review"]["verdict"] != "PASS":
            raise ValueError("Both independent critics must pass")
        stages.append((stage, review["review"], review["receipt"]))
    if verify_receipts:
        from .client import validate_receipt

        if len({(r["journal"], r["byte_offset"]) for _, _, r in stages}) != 4:
            raise ValueError("The four stages require distinct request receipts")
        for stage, value, receipt in stages:
            validate_receipt(receipt, stage, frozen, value, body=body,
                             observer=observer, style=annotation["style_name"],
                             model=annotation["model"], row_index=annotation["row_index"],
                             journal_root=journal_root)
    return frozen
