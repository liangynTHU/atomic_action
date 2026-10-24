"""Bounded, journaled teacher requests; only accepted labels are cached."""

import json
import time
from pathlib import Path

import numpy as np

from atomic_seg.common.model_client import multimodal_content
from atomic_seg.pretrain.pipeline import validate_labels
from atomic_seg.pretrain.length_policy import validate_action_primitives
from atomic_seg.pretrain.prompts import ACTION_STARTS
from atomic_seg.pretrain.realomni.storage import Journal, atomic_json, digest, file_hash, read_json
from .contract import teacher_system, teacher_review_system
from .grounding import grounding_facts, grounding_system, source_phase_hints, validate_grounding
from .media import sample_frames, temporal_camera_sheet
from .api import ServiceUnavailable, chat_url, final_json, generation_parameters, request_json, response_audit


class AnnotationRejected(ValueError):
    pass


def per_arm_observations(episode, segment, frames):
    states = np.asarray(episode["measured_states"], dtype=float)
    result = {}
    for arm, offset in (("left", 0), ("right", 8)):
        sampled = states[frames, offset:offset + 8]
        trajectory = states[segment["start_frame"]:segment["end_frame"] + 2, offset:offset + 8]
        result["physical_" + arm] = {
            "wrist_camera": "hand_" + arm,
            "action": segment[arm + "_action"],
            "required_clause_start": ACTION_STARTS[segment[arm + "_action"]],
            "observations": [
                {"frame": frame, "xyz_m": state[:3].tolist(),
                 "quat_wxyz": state[3:7].tolist(), "encoder_openness_proxy": float(state[7])}
                for frame, state in zip(frames, sampled)
            ],
            "openness_range": [float(trajectory[:, 7].min()), float(trajectory[:, 7].max())],
            "position_delta_m": (trajectory[-1, :3] - trajectory[0, :3]).tolist(),
            "position_path_m": float(np.linalg.norm(np.diff(trajectory[:, :3], axis=0), axis=1).sum()),
        }
    return result


def visual_boundary(segment, check):
    return {"frame": segment["end_frame"] + 1, "segment_index": segment["segment_index"],
            **{arm: {"state": check[arm]["target_state"], "object": check[arm]["object"]
                     if check[arm]["target_state"] == "held" else None}
               for arm in ("left", "right")},
            "authority": "previous model observation, not ground truth; current identical-frame images are authoritative"}


def adjudicated_boundary(boundary, adjudications):
    """Apply an explicit, evidenced correction to a previous model boundary."""
    decision = (adjudications or {}).get(str(boundary["frame"]))
    if decision is None:
        return boundary
    if set(decision) != {"frame", "evidence", "arms"} or decision["frame"] != boundary["frame"]:
        raise ValueError("boundary adjudication must identify its frame, evidence and arms")
    if not isinstance(decision["evidence"], str) or not decision["evidence"].strip():
        raise ValueError("boundary adjudication needs an evidence note")
    if not isinstance(decision["arms"], dict) or not decision["arms"] or set(decision["arms"]) - {"left", "right"}:
        raise ValueError("boundary adjudication needs one or both physical arms")
    result = dict(boundary)
    for arm, state in decision["arms"].items():
        if state not in {"empty", "unknown"}:
            raise ValueError("boundary adjudication only supports empty or unknown")
        result[arm] = {"state": state, "object": None}
    result["authority"] = "explicit boundary adjudication from supplied images; earlier model observation preserved in raw review"
    result["adjudication"] = decision["evidence"]
    return result


def validate_visual_review(value, segments, previous_boundary=None):
    if set(value) != {"segments", "visual_checks"}:
        raise ValueError("visual review requires exactly segments and visual_checks")
    validate_labels({"segments": value["segments"]}, segments)
    checks = value["visual_checks"]
    if not isinstance(checks, list) or len(checks) != len(segments):
        raise ValueError("missing visual checks")
    for check, segment in zip(checks, segments):
        if set(check) != {"segment_index", "left", "right"} or check["segment_index"] != segment["segment_index"]:
            raise ValueError("visual check indices must match requested order")
        frames = set(sample_frames([segment]))
        for arm in ("left", "right"):
            state = check[arm]
            if set(state) != {"start_state", "target_state", "object", "evidence_frames"}:
                raise ValueError("visual check must include both boundary states, object and evidence frames")
            if any(state[name] not in {"held", "empty", "unknown"} for name in ("start_state", "target_state")):
                raise ValueError("visual state must be held, empty or unknown")
            obj = state["object"]
            if obj is not None and (not isinstance(obj, str) or not obj.strip() or len(obj) > 120):
                raise ValueError("visual object must be a short name or null")
            if "held" in (state["start_state"], state["target_state"]) and obj is None:
                raise ValueError("held state needs a visible object name")
            if state["start_state"] == state["target_state"] == "empty" and obj is not None:
                raise ValueError("empty boundary states must not name a held object; use null")
            evidence = state["evidence_frames"]
            if (not isinstance(evidence, list) or not evidence
                    or any(type(frame) is not int or frame not in frames for frame in evidence)):
                raise ValueError("visual evidence must reference supplied interval frames")
        if previous_boundary is not None and previous_boundary["frame"] == segment["start_frame"]:
            for arm in ("left", "right"):
                before, now = previous_boundary[arm]["state"], check[arm]["start_state"]
                if before != "unknown" and now != "unknown" and before != now:
                    raise ValueError(
                        f"Shared-frame contradiction at frame {segment['start_frame']}, physical {arm}: "
                        f"previous target={before}, current start={now}. These are the SAME observation, "
                        "not a grasp/release or handover across time. Reinspect this arm's own fingertips "
                        "in all views, correct the state AND affected labels. If evidence is genuinely "
                        "ambiguous, use unknown and omit unsupported holding/release claims.")
        previous_boundary = visual_boundary(segment, check)


def accepted_request(cache, failure, system, text, images, roles, parameters, endpoints,
                     api_key, journal, batch_index, settings, validator, *, stage):
    image_ids = [{"sha256": file_hash(p), "role": role} for p, role in zip(images, roles)]
    identity = digest({"system": system, "user": text, "images": image_ids,
                       "parameters": parameters, "stage": stage})
    if cache.exists():
        accepted = read_json(cache)
        if accepted["identity"] != identity:
            raise RuntimeError(f"{stage} cache contract changed")
        validator(accepted["labels"])
        return accepted
    previous = read_json(failure) if failure.exists() else {"identity": identity, "content_failures": 0}
    if previous["identity"] != identity:
        raise RuntimeError(f"{stage} attempt contract changed")
    failures, feedback = previous["content_failures"], previous.get("last_error", "")
    if not endpoints or not api_key:
        raise ServiceUnavailable("API endpoint and token required")
    while failures < settings.semantic_attempts:
        payload = {**parameters, "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": multimodal_content(
                text + ("\nPrevious validation error: " + feedback if feedback else ""), images, roles)}]}
        response, errors = None, []
        for retry in range(3):
            endpoint = endpoints[(batch_index + failures + retry) % len(endpoints)]
            begun = time.time()
            journal.append({"event": "request", "identity": identity, "batch": batch_index, "stage": stage,
                            "endpoint": endpoint, "system": system, "user": text, "feedback": feedback,
                            "images": image_ids, "parameters": parameters, "started_at": begun})
            try:
                response = request_json(chat_url(endpoint), api_key, payload=payload)
                journal.append({"event": "response", "identity": identity, "batch": batch_index, "stage": stage,
                                "response": response_audit(response), "elapsed_s": time.time() - begun})
                break
            except ServiceUnavailable as error:
                errors.append(str(error))
                journal.append({"event": "service_error", "identity": identity, "stage": stage,
                                "error": str(error), "elapsed_s": time.time() - begun})
                if retry < 2:
                    time.sleep(min(30, 5 * 2**retry))
        if response is None:
            raise ServiceUnavailable("; ".join(errors))
        try:
            labels = final_json(response)
            validator(labels)
            accepted = {"identity": identity, "labels": labels}
            atomic_json(cache, accepted)
            return accepted
        except (ValueError, KeyError, TypeError, IndexError, AttributeError) as error:
            failures += 1
            feedback = str(error)
            atomic_json(failure, {"identity": identity, "content_failures": failures, "last_error": feedback})
    raise AnnotationRejected(f"{stage} batch {batch_index}: {feedback}")


def annotate(episode, segmentation, media_dir, directory, settings, endpoints, checkpoint,
             *, api_key=None, boundary_adjudications=None):
    for segment in segmentation["segments"]:
        validate_action_primitives(segment)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    journal = Journal(directory / "requests.jsonl")
    output = [dict(s) for s in segmentation["segments"]]
    completed = 0
    previous_boundary = None
    for batch_index, start in enumerate(range(0, len(output), settings.batch_size)):
        segments = output[start:start+settings.batch_size]
        images, roles, facts = [], [], []
        history = None
        if start:
            previous = output[start - 1]
            previous_frames = sample_frames([previous])
            for frame in previous_frames:
                images.append(Path(media_dir) / "teacher" / f"{frame:07d}.jpg")
                roles.append(f"PAST CONTEXT segment {previous['segment_index']} observation {frame}; "
                             "panels head | physical LEFT wrist | physical RIGHT wrist; do not label")
            # Include the entire last transition, not just its potentially pre-release start.
            history = {"segment_index": previous["segment_index"],
                       "start_frame": previous["start_frame"],
                       "target_frame": previous["end_frame"] + 1,
                       "sampled_observations": previous_frames,
                       "per_arm_observations": per_arm_observations(episode, previous, previous_frames),
                       "role": "completed past interval, not requested output; its target is the CURRENT batch start"}
        for s in segments:
            frames = sample_frames([s])
            for frame in frames:
                images.append(Path(media_dir) / "teacher" / f"{frame:07d}.jpg")
                roles.append(f"CURRENT REQUEST segment {s['segment_index']} measured observation {frame}; "
                             "panels head | physical LEFT wrist | physical RIGHT wrist")
            facts.append({**s, "sampled_observations": frames,
                          **({"source_phase_hints": source_phase_hints(episode, s)}
                             if settings.object_grounding else {}),
                          "per_arm_observations": per_arm_observations(episode, s, frames),
                          "required_clause_starts": {arm: ACTION_STARTS[s[arm+"_action"]] for arm in ("left", "right")},
                          "measured_start_state": episode["measured_states"][s["start_frame"]],
                          "measured_target_state": episode["measured_states"][s["end_frame"]+1]})
        system = teacher_system(settings)
        parameters = generation_parameters(settings)
        image_ids = [{"sha256": file_hash(p), "role": role} for p, role in zip(images, roles)]
        review_images, review_roles = [], []
        for s in ([history] if history else []) + facts:
            frames = s["sampled_observations"]
            for camera in ("head", "hand_left", "hand_right"):
                path = directory / "visual_evidence" / f"segment_{s['segment_index']:05d}_{camera}.jpg"
                temporal_camera_sheet(media_dir, camera, frames, path)
                review_images.append(path)
                review_roles.append(f"{'PAST CONTEXT' if s is history else 'REQUESTED'} "
                                    f"segment {s['segment_index']}; {camera}; chronological frames {frames}")
            if s is not history:
                for frame in (frames[0], frames[-1]):
                    review_images.append(Path(media_dir) / "head" / f"{frame:07d}.jpg")
                    review_roles.append(f"REQUESTED segment {s['segment_index']} full HEAD boundary frame {frame}; "
                                        "verify the WHOLE object shape and both physical arms, not only handle color")
        review_image_ids = [{"sha256": file_hash(p), "role": role}
                            for p, role in zip(review_images, review_roles)]
        object_grounding = None
        if settings.object_grounding:
            grounded = accepted_request(
                directory / f"batch_{batch_index:05d}.objects.json",
                directory / f"batch_{batch_index:05d}.objects.attempts.json",
                grounding_system(),
                json.dumps({"task_context_not_observed_state": episode["task"],
                            "requested_intervals": grounding_facts(episode, segments),
                            "camera_panels": ["head", "physical_left_wrist_camera",
                                              "physical_right_wrist_camera"]}, ensure_ascii=False),
                review_images, review_roles, parameters, endpoints, api_key, journal,
                batch_index, settings, lambda value: validate_grounding(value, segments),
                stage="object_grounding")
            # Atomic JSON storage sorts keys. Restore the request's field order
            # so a resumed batch has the same prompt bytes and cache identity.
            object_grounding = [
                {"segment_index": item["segment_index"], **{
                    arm: {key: item[arm][key] for key in
                          ("name", "ref_phrase", "relation", "evidence_frames")}
                    for arm in ("left", "right")}}
                for item in grounded["labels"]["objects"]
            ]
        request_facts = {"task_context_not_observed_state": episode["task"],
                         "segments": facts, "completed_previous_interval": history,
                         "previous_visual_boundary": previous_boundary,
                         "camera_panels": ["head", "physical_left_wrist_camera", "physical_right_wrist_camera"],
                         "evidence_rule": "Observe each physical gripper and object at CURRENT interval start and target, then use the middle observations for co-motion. Named measurements determine fixed motion verbs, not object ownership or visible finger pose. Earlier images and task wording cannot override current empty-gripper evidence. Opening an empty gripper is not releasing an object; do not claim historical release without seeing a held object beforehand.",
                         "response_rule": "Return the JSON object directly, without Markdown fences. Each sub_task MUST literally contain 'Left arm: ' and '. Right arm: ', using each required verb. Example: Left arm: Hold the current pose. Right arm: Move toward the visible object. Label only requested indices and the actual phase in their own images. Keep supported object names and destinations; omit only uncertain details. Do not describe a future grasp or an already completed release. Use neutral movement direction if ambiguous."}
        if object_grounding is not None:
            request_facts["object_grounding"] = object_grounding
        text = json.dumps(request_facts, ensure_ascii=False)
        identity = digest({"system": system, "user": text, "images": image_ids,
                           "parameters": parameters, "review_system": teacher_review_system(settings.object_grounding),
                           "review_images": review_image_ids, "previous_visual_boundary": previous_boundary})
        cache = directory / f"batch_{batch_index:05d}.json"
        if cache.exists():
            accepted = read_json(cache)
            if accepted["identity"] != identity:
                raise RuntimeError("annotation cache contract changed")
            validate_labels(accepted["labels"], segments)
            validate_visual_review({"segments": accepted["labels"]["segments"],
                                    "visual_checks": accepted["visual_checks"]}, segments, previous_boundary)
        else:
            candidate = accepted_request(
                directory / f"batch_{batch_index:05d}.candidate.json",
                directory / f"batch_{batch_index:05d}.attempts.json", system, text, images, roles,
                parameters, endpoints, api_key, journal, batch_index, settings,
                lambda value: validate_labels(value, segments), stage="candidate")
            review_facts = {"segments": facts, "completed_previous_interval": history,
                            "previous_visual_boundary": previous_boundary,
                            "candidate_labels": candidate["labels"]["segments"],
                            "output_template": {
                                          "segments": candidate["labels"]["segments"],
                                          "visual_checks": [
                                              {"segment_index": s["segment_index"], **{
                                                  arm: {"start_state": "unknown", "target_state": "unknown",
                                                        "object": None, "evidence_frames": [
                                                            s["start_frame"], s["end_frame"] + 1]}
                                                  for arm in ("left", "right")}}
                                              for s in segments]},
                            "response_rule": "Independently inspect each named camera's own near-field gripper. Return corrected segments and per-arm visual_checks for requested indices only."}
            if object_grounding is not None:
                review_facts["task_context_not_observed_state"] = episode["task"]
                review_facts["object_grounding"] = object_grounding
            review_text = json.dumps(review_facts, ensure_ascii=False)
            reviewed = accepted_request(
                directory / f"batch_{batch_index:05d}.review.json",
                directory / f"batch_{batch_index:05d}.review.attempts.json",
                teacher_review_system(settings.object_grounding), review_text, review_images, review_roles,
                parameters, endpoints, api_key, journal, batch_index, settings,
                lambda value: validate_visual_review(value, segments, previous_boundary), stage="visual_review")
            accepted = {"identity": identity, "labels": {"segments": reviewed["labels"]["segments"]},
                        "visual_checks": reviewed["labels"]["visual_checks"],
                        "candidate_identity": candidate["identity"], "review_identity": reviewed["identity"]}
            atomic_json(cache, accepted)
        previous_boundary = adjudicated_boundary(
            visual_boundary(segments[-1], accepted["visual_checks"][-1]), boundary_adjudications)
        for segment, label in zip(segments, accepted["labels"]["segments"]):
            segment["sub_task"] = label["sub_task"]
        completed += len(segments)
        checkpoint(completed, len(output))
    return output
