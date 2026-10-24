"""Image-first object hypotheses for an optional AgiBot annotation experiment."""

import numpy as np


def sample_frames(segments):
    return sorted({int(index) for segment in segments
                   for index in np.linspace(segment["start_frame"], segment["end_frame"] + 1,
                                            5, dtype=int)})


RELATIONS = {"held", "contacting", "approaching", "near", "none", "unclear"}


def source_phase_hints(episode, segment):
    """Return only coarse source actions overlapping this fixed interval."""
    start, target = segment["start_frame"], segment["end_frame"] + 1
    return [
        {"start_frame": phase["start_frame"], "end_frame": phase["end_frame"],
         "action_text": phase["text"], "skill": phase.get("skill")}
        for phase in episode.get("source_annotations", [])
        if phase["start_frame"] < target and phase["end_frame"] > start
    ]


def grounding_facts(episode, segments):
    return [
        {"segment_index": segment["segment_index"],
         "start_frame": segment["start_frame"],
         "target_frame": segment["end_frame"] + 1,
         "sampled_observations": sample_frames([segment]),
         "physical_left_action": segment["left_action"],
         "physical_right_action": segment["right_action"],
         "source_phase_hints": source_phase_hints(episode, segment)}
        for segment in segments
    ]


def grounding_system():
    return (
        "Resolve object identity for fixed robot-motion intervals before writing any sub_task. "
        "Each requested interval has chronological head, physical-left-wrist and physical-right-wrist "
        "images. A wrist camera is a local view, not proof that everything in it belongs to that arm; "
        "the full head view establishes scene context. Inspect the whole object, its handle and working "
        "end, its parent or support, and whether the whole object moves with the physical gripper. "
        "Distinguish a removable tool from its stationary support, a rigid container from its liner, "
        "and a moving hinged part from its stationary parent when the images support this. "
        "The task and source_phase_hints are weak, possibly wrong descriptions of intended activity. "
        "They are not evidence of current contact, grasp, placement, or success. Use them to locate a "
        "possible target, then correct the name if the images disagree. "
        "Do not copy a source name when the object is visually different. "
        "An encoder value or a closed gripper is not proof "
        "of holding an object. Base held/contacting on visible contact or object-gripper co-motion. "
        "For each physical arm in each requested interval, return the principal visible object it "
        "interacts with or holds. If there is no identifiable object, use null for name and "
        "ref_phrase and relation none or unclear. If a formerly held object remains visible "
        "after release, keep its name and ref_phrase with relation none. An object near but "
        "not held uses near. "
        "name is an open-vocabulary object or object-part noun phrase. ref_phrase identifies that "
        "specific instance in the full scene by appearance, location or relation to another object. "
        "Do not force an object into a fixed vocabulary and do not invent unseen features. "
        "evidence_frames must cite supplied observations of that same interval. "
        "Return JSON only with exactly one objects list. Each item has exactly segment_index, left, "
        "right; each arm has exactly name, ref_phrase, relation, evidence_frames. "
        "relation must be held, contacting, approaching, near, none, or unclear. "
        'Example shape: {"objects":[{"segment_index":0,"left":{"name":null,'
        '"ref_phrase":null,"relation":"none","evidence_frames":[0,8]},'
        '"right":{"name":"handheld tool","ref_phrase":"white handheld tool in the stand",'
        '"relation":"approaching","evidence_frames":[0,8]}}]}. '
        "Use actual supplied indices and frames; no sub_task, reasoning or extra keys."
    )


def validate_grounding(value, segments):
    if not isinstance(value, dict) or set(value) != {"objects"} or not isinstance(value["objects"], list):
        raise ValueError("object grounding requires exactly an objects list")
    if len(value["objects"]) != len(segments):
        raise ValueError("object grounding must cover all requested segments")
    for item, segment in zip(value["objects"], segments):
        if not isinstance(item, dict) or set(item) != {"segment_index", "left", "right"}:
            raise ValueError("object grounding item has wrong keys")
        if item["segment_index"] != segment["segment_index"]:
            raise ValueError("object grounding indices must match requested order")
        frames = set(sample_frames([segment]))
        for arm in ("left", "right"):
            record = item[arm]
            if not isinstance(record, dict) or set(record) != {"name", "ref_phrase", "relation", "evidence_frames"}:
                raise ValueError("object grounding arm has wrong keys")
            name, ref, relation = record["name"], record["ref_phrase"], record["relation"]
            if relation not in RELATIONS:
                raise ValueError("invalid object relation")
            if name is None or ref is None:
                if name is not None or ref is not None or relation not in {"none", "unclear"}:
                    raise ValueError("null object requires both names null and relation none or unclear")
            elif (not isinstance(name, str) or not name.strip() or len(name) > 100
                  or not isinstance(ref, str) or not ref.strip() or len(ref.split()) > 24):
                raise ValueError("visible object needs a short name and a specific ref_phrase")
            cited = record["evidence_frames"]
            if (not isinstance(cited, list) or not cited
                    or any(type(frame) is not int or frame not in frames for frame in cited)):
                raise ValueError("object evidence must cite supplied interval frames")
