"""16D successor-state supervision and AgiBot-specific language."""

import math
import numpy as np

from atomic_seg.prompt_contract import (
    NEXT_ATOMIC_SUBTASK, RECOMPUTE_GUIDE, CHUNK_COUNT_CONTRACT,
    MULTI_ARM_CONTRACT, PREVIOUS_COMMAND_CONTRACT, NO_THINKING_OUTPUT_CONTRACT,
    NO_THINKING_USER_ENDING,
)
from atomic_seg.pretrain.realomni.reference import reference_episode, segment_reference
from atomic_seg.pretrain.student_format import named_state, json_text
from atomic_seg.pretrain.segmentation import target_for
from atomic_seg.pretrain.prompts import SAME_ACTION_GROUP_CONTRACT


def action_contract(settings):
    return {
        "dimension": 16, "robot_type": "a2d", "robot": "AgiBot dual-arm mobile robot (dataset type a2d)",
        "state_layout": [f"{side}_{x}" for side in ("left", "right")
                         for x in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")],
        "reference_action": "action[t] = measured_state[t+1]",
        "guide_action": "action[end] - state[start] = measured_state[end+1] - measured_state[start]",
        "quaternion": "wxyz; normalized sign-continuous states; guide is component difference",
        "coordinate_frame": "recorded AgiBot end/flange frame; no invented world-frame transform",
        "gripper": {"quantity": "normalized measured encoder openness proxy, not physical aperture in mm",
                    "formula": "(closed_anchor - measured_encoder) / (closed_anchor - open_anchor)",
                    "open_anchor": settings.gripper_open, "closed_anchor": settings.gripper_closed,
                    "normalization": "fixed dataset-level anchors; no per-episode rescaling or clipping",
                    "calibration_status": "approximate encoder representation anchors, visually validated; not physical jaw-width calibration",
                    "direction": "approximately 1=open, 0=closed; small measurement overshoot retained"},
        "num_chunks": "ceil(valid_action_rows/8)",
        "terminal": "N measurements yield N-1 transitions; final observation retained",
        "native_actions": "preserved separately; never used as reference end targets",
    }


def build_reference(arrays, timestamps_ns, settings, *, episode_id, task, source):
    position = arrays["observation.states.end.position"]
    quat = arrays["observation.states.end.orientation"][..., [3, 0, 1, 2]]
    grip = (settings.gripper_closed - arrays["observation.states.effector.position"]) / (
        settings.gripper_closed - settings.gripper_open)
    measured = np.concatenate([np.column_stack((position[:, arm], quat[:, arm], grip[:, arm]))
                               for arm in (0, 1)], axis=1)
    # Permit only bounded floating-point norm errors, then normalize before
    # entering the common measured-successor implementation.
    for offset in (3, 11):
        measured[:, offset:offset+4] /= np.linalg.norm(measured[:, offset:offset+4], axis=1)[:, None]
    episode = reference_episode(measured, timestamps_ns, task=task, episode_id=episode_id, source=source)
    episode.update(dataset="Agibot", subset="agibot_beta_v3", profile="agibot_a2d",
                   action_contract=action_contract(settings),
                   time_basis="LeRobot episode frame clock at 30 Hz; retained source float timestamps in native arrays")
    return episode


def segment(episode, settings):
    result = segment_reference(episode, length_policy=settings.length_policy)
    result["action_contract"] = action_contract(settings)
    return result


def student_system(settings):
    c = action_contract(settings)
    return (
        "You are the high-level planner for an AgiBot dual-arm robot (dataset robot_type a2d). "
        + NEXT_ATOMIC_SUBTASK
        + "The 16-D state and guide_action use this order: [" + ",".join(c["state_layout"]) + "]. "
        "Positions are measured flange/end positions in meters in the recorded AgiBot coordinate frame. "
        "Orientation arrays are named quat_wxyz and use scalar-first unit quaternions. "
        f"Gripper is the measured encoder openness proxy ({settings.gripper_closed:g}-raw)/"
        f"({settings.gripper_closed:g}-{settings.gripper_open:g}), approximately 1=open and 0=closed. "
        "It is not a jaw width in millimeters; small measurement overshoot is retained. "
        "Reference action[t] is the absolute measured end-effector state[t+1], not an original controller command. "
        "guide_action is the remaining component-wise 16-D residual target_endpoint_state-current_origin_state. "
        "The endpoint is measured_state[end+1], following the final inclusive reference action. "
        "Quaternion guide entries are component differences between sign-continuous quaternions, not a relative rotation. "
        "Zero gripper delta means unchanged. " + RECOMPUTE_GUIDE + CHUNK_COUNT_CONTRACT
        + "Episodes with moving or unverifiable chassis are excluded. Left/right identify physical arms, not screen sides. "
        + MULTI_ARM_CONTRACT + PREVIOUS_COMMAND_CONTRACT + NO_THINKING_OUTPUT_CONTRACT
    )


def teacher_system(settings):
    base = (
        "Label fixed atomic motion intervals from the AgiBot dual-arm robot (dataset type a2d). "
        "The state is 16D: left xyz, quaternion wxyz, measured gripper openness, then right in the same order. "
        "XYZ is in meters in the recorded robot end/flange frame; rotations are sign-continuous unit quaternions. "
        "Gripper decreases on closing and increases on opening, approximately 0=closed and 1=open. "
        "It is an approximate measured encoder proxy, not evidence of object contact or physical jaw width. "
        "Reference action[t]=measured_state[t+1]; numerical targets and boundaries are already fixed. "
        "Each interval includes five distinct chronological observations when available, including the target observation "
        "after its last action. Each montage has three panels in fixed order: head, hand_left, hand_right. "
        "ARM IDENTITY: Left and right ALWAYS mean the robot's physical arms, never screen-left/screen-right. "
        "hand_left is the camera mounted on the PHYSICAL LEFT wrist; hand_right is mounted on the PHYSICAL RIGHT wrist. "
        "A wrist camera can see BOTH grippers: its own nearby gripper and the OTHER arm farther away. "
        "Do NOT assign every gripper or object visible in hand_right to the right arm, or vice versa. "
        "Ground each physical arm using its wrist-mounted near-field gripper, the head view, and chronological motion "
        "across views. per_arm_observations is explicitly keyed by physical_left and physical_right. "
        "Its action and required_clause_start belong only to that physical arm; never exchange them. "
        "OBJECT OWNERSHIP: Track which physical gripper actually holds each object from interval start to end. "
        "A closed or closing encoder value alone does NOT establish a grasp. Objects beside, behind, or touched "
        "by the outside of a gripper are not necessarily held. Require visible grasp/contact geometry or consistent "
        "object-gripper co-motion before writing held, gripped, grasped, carrying, or securing. "
        "Keep the same object's arm identity across intervals unless the images show a real transfer, release, "
        "or new grasp. A changed camera angle or batch boundary is NOT a handover. "
        "Optional completed_previous_interval images show ALL observations of the last completed interval, "
        "including its target, which equals the current batch start. They are context only, not output indices. "
        "previous_visual_boundary records the prior review's physical-arm state at that exact shared frame. "
        "It is a model observation, not ground truth; compare it against the current identical-frame image. "
        "An empty prior boundary is not a held object merely because a new batch starts. "
        "OBJECT IDENTITY: Inspect the complete visible object, not just its handle color or a partial silhouette. "
        "A colored handle attached to a metal blade does not by itself imply a spatula. Prefer the most specific "
        "category supported by geometry and current interaction; if tool type is ambiguous use 'tool' rather "
        "than inventing another category. Camera changes alone do not change a held object's identity. "
        "TEMPORAL PRECEDENCE: For every requested interval, inspect the CURRENT START and TARGET grippers "
        "first, then the middle images. An earlier held object does not imply it is held now. "
        "After visible release, ownership is NONE until a visible new grasp; after a failed or empty close, "
        "ownership remains NONE. Do not restart the task narrative at a batch boundary. "
        "INTERVAL PHASE: Distinguish approaching an object from grasping it, and releasing it from withdrawing "
        "an empty gripper. If grasp occurs in the next interval, this interval only approaches or aligns. "
        "If an object already rests on a support and the open gripper moves away, describe the empty gripper, "
        "not moving, rotating, placing, or releasing that object again. Rotate may describe the wrist alone; "
        "it does not imply rotating a nearby object. Hold means maintaining that arm's pose and does not imply "
        "holding an object. Use all five observations; do not extrapolate a coarse task phase into a later interval. "
        "ACTIVITY EVIDENCE: Carrying a cloth toward/away from a surface is not wiping it. A knife above food "
        "is not yet cutting. Name functional activities only when the current images show sustained relevant "
        "contact and motion; otherwise describe the visible transport, approach or withdrawal. Do not infer "
        "an intention such as 'to place', 'to wipe' or 'to grasp' from task wording or object proximity alone. "
        "Open the gripper means finger opening, not automatically releasing an object. To say release, "
        "the SAME arm must visibly hold the SAME object BEFORE opening. If empty beforehand, say only "
        "Open the gripper. Do not add 'after releasing' without visual evidence of that past event. "
        "EVIDENCE CONFLICT: Preserve the fixed required verb even if encoder openness and visible fingers disagree; "
        "describe only the gripper motion without inventing an object, contact, achieved grasp, or achieved release. "
        "When object ownership, direction, or identity is uncertain, omit that claim and use a neutral description "
        "of the correctly identified physical arm with its required verb. Never resolve uncertainty by swapping arms. "
        "Static finger descriptions such as open or closed require visible geometry; encoder values do not "
        "license them. A Hold clause may simply preserve the pose without a finger-state adjective. "
        "Retain clearly visible objects, tools, surfaces and movement context; do not make every clause generic. "
        "The student sees only current/past images. Future evidence is only for offline annotation. "
        "Describe the whole supplied interval using primitive_segments and motion_sequence. "
        "Use exactly 'Left arm: <imperative>. Right arm: <imperative>.' Each clause must begin with its "
        "required_clause_starts verb. Hold means hold that arm, not task completion. Preserve required gripper actions. "
        "Task wording is context, not a claim about current ownership, successful execution or next phase. "
        "Never invent contact, grasp security, success, occluded objects or a future phase. "
        "Before returning, silently check BOTH clauses against their own physical gripper, object ownership, "
        "and interval start/end phase: empty/held at start, empty/held at target, whether contact/co-motion "
        "supports the named object, and whether any release belongs to this interval or the past. "
        "Do not expose this check or chain of thought, and never include audit instructions in sub_task. "
        + SAME_ACTION_GROUP_CONTRACT
        + "Each label is at most 70 English words. "
        'Return JSON only: {"segments":[{"segment_index":0,"sub_task":"..."}]}. '
        "Return all requested indices once in order. No other keys, reasoning, thought tags, numeric actions or boundaries."
    )
    if not settings.object_grounding:
        return base
    return base + (
        " EXPERIMENTAL OBJECT GROUNDING: An earlier image-first object pass supplies "
        "object_grounding for the requested intervals. It has seen the task and coarse source "
        "phase hints, but these are weak clues, not observed outcomes. Use each arm's grounded "
        "open-vocabulary name and scene-specific ref_phrase when supported by this interval's images. "
        "Resolve removable tool versus fixed support, rigid container versus liner, and moving "
        "part versus stationary parent before naming an object in sub_task. The grounding record "
        "is a model hypothesis, not ground truth: correct it if current images contradict it, "
        "and omit unsupported contact, holding or future results. Give the physical arm, "
        "motion, specific visible object or part, and visible location in plain imperative language."
    )


def build_user(episode, current, previous, camera_names):
    frames = ([] if previous is None else [("previous-boundary", previous["start_frame"])]) + [("current", current["start_frame"])]
    roles = [f"{when} {name}" for when, _ in frames for name in camera_names]
    lines = ["<image>" * len(roles) + "Task: " + episode["task"],
             "Task arm mode: multi-arm; both arms may participate or hand off as required by the task.",
             "Images: " + ", ".join(roles) + "."]
    state = lambda x, delta=False: named_state("realomni_umi", x, delta=delta)
    start = current["start_frame"]
    if previous is None:
        lines.append("History boundary: none; this is the initial planning boundary.")
    else:
        if previous["end_frame"] + 1 != start:
            raise ValueError("noncontiguous previous decision")
        origin = previous["start_frame"]
        n = start - origin
        lines += ["Previous command: " + json_text(target_for(episode, previous)),
                  "Previous execution: " + json_text({"executed_chunks": math.ceil(n/8), "valid_action_rows": n,
                      "aligned_action_slots": math.ceil(n/8)*8, "padded_action_slots": (-n) % 8}),
                  "Previous boundary state: " + json_text(state(episode["states"][origin]))]
    lines.append("Current state: " + json_text(state(episode["states"][start])))
    if previous is not None:
        lines.append("Observed state delta since previous boundary: " + json_text(
            state(np.asarray(episode["states"][start]) - episode["states"][origin], delta=True)))
    lines.append(NO_THINKING_USER_ENDING)
    return "\n".join(lines), frames, roles


def teacher_review_system(object_grounding=False):
    base = (
        "Independently verify and correct candidate AgiBot atomic sub_task labels using the supplied images. "
        "Candidate labels are untrusted guesses, NOT evidence. No task narrative or source phase is supplied. "
        "Named-camera contact sheets contain up to five observations in chronological READING ORDER: "
        "left-to-right on the first row, then left-to-right on the second row. Frames are explicitly named. "
        "Each sheet is ONE camera: HEAD, PHYSICAL LEFT WRIST, or PHYSICAL RIGHT WRIST. Additional full HEAD "
        "boundary images preserve whole-object details; inspect them before deciding an object category. "
        "Identify the camera-mounted near-field gripper at the bottom foreground of each wrist tile. "
        "A tool held by the OTHER arm farther away in that image is NOT held by this camera's own gripper. "
        "For EACH physical arm at EACH interval, observe its OWN fingertips at START and TARGET first. "
        "Record held, empty or unknown and the object's generic visible name, supported by listed image frames. "
        "Then inspect intermediate co-motion and head view to verify approach, contact, grasp or release. "
        "A bottle/tool behind or beside the fingers is not a grasp; empty closing stays empty. "
        "OBJECT CATEGORY: Inspect the whole object across views, including its working end. Handle color "
        "alone cannot distinguish a knife from a spatula or a pump bottle from a spray bottle. Do not replace "
        "a candidate's supported generic object name with a more speculative type. If the image cannot "
        "resolve the type, use a neutral visible category such as tool or bottle. Keep the same visible "
        "object's category across adjacent intervals unless clear geometry disproves it. "
        "If a released object stays on a support while the gripper retreats, describe the empty wrist/gripper. "
        "Opening an empty gripper is not release. Never invent a transfer between arms or a past release. "
        "SHARED FRAME CONTINUITY: An interval's target frame is the next interval's start frame: the SAME image. "
        "Its physical-arm held/empty states cannot disagree across those two labels, including across batches. "
        "previous_visual_boundary is the previous review's recorded state at that shared frame, NOT ground truth. "
        "Reinspect the identical image against that record. Correct unsupported claims; when genuinely occluded "
        "use unknown instead of confidently inventing a held object. Never force an uncertain prior guess. "
        "Within the requested batch, make all adjacent target/start states consistent too. A real grasp or release "
        "can change state DURING an interval, not between two descriptions of an identical frame. "
        "A release claim needs this SAME arm visibly holding the SAME object earlier in the provided timeline. "
        "If it was already empty at the shared start and remains empty, remove both 'release' and 'after releasing' "
        "unless intermediate images actually show a new grasp followed by release. "
        "A cloth moving off a surface toward another location is transport, not wiping that surface. "
        "Functional activities and purposes require visible contact/motion evidence in THIS interval; proximity "
        "or a held tool alone is insufficient. Prefer actual movement over inferred 'to wipe/place/grasp'. "
        "Measurements authorize only the supplied required motion verb, not object ownership or finger pose. "
        "Hold means stationary arm; Rotate may rotate just the wrist. Preserve each physical arm's required verb. "
        "If ownership/contact is unclear, remove only the unsupported claim; retain visible objects and context "
        "where supported. Do not claim completion, future events or task success. "
        "Return corrected labels, even if a candidate was plausible; keep a correct candidate unchanged. "
        "Each sub_task uses exactly 'Left arm: <imperative>. Right arm: <imperative>.' with each required "
        "clause-start verb and at most 70 English words. Do not change numeric actions or interval boundaries. "
        "Output JSON only with exactly segments and visual_checks. Follow output_template exactly for array "
        "lengths, key placement and JSON punctuation; replace its values with inspected results. "
        "Use no extra closing brackets, trailing commas, Markdown fences or surrounding prose. "
        'Schema: {"segments":[{"segment_index":0,"sub_task":"..."}],'
        '"visual_checks":[{"segment_index":0,"left":{"start_state":"held|empty|unknown",'
        '"target_state":"held|empty|unknown","object":"generic visible name or null",'
        '"evidence_frames":[0,8]},"right":{"start_state":"held|empty|unknown",'
        '"target_state":"held|empty|unknown","object":"generic visible name or null",'
        '"evidence_frames":[0,8]}}]}. '
        "Use actual state enum values, not the pipe-separated schema strings. "
        "object names the held object; use null when both boundary states are empty. "
        "Both arrays must have only all requested indices, once in order. Evidence frames must be supplied "
        "observations of that requested interval. No reasoning, explanations or audit instructions in sub_task."
    )
    if not object_grounding:
        return base
    return base + (
        " EXPERIMENTAL OBJECT GROUNDING: You also receive the original task, overlapping "
        "coarse source phase hints, and an image-first object_grounding ledger. Treat every "
        "one as a fallible clue, not proof of the present action or grasp. Inspect full head "
        "boundary views and chronological wrist views yourself. If a candidate confuses a "
        "removable tool with its fixed stand, a trash can with its liner, or a hinged lid "
        "with its stationary parent, correct the object name and the action description. "
        "Keep a supported specific name and ref_phrase in the corrected sub_task; use a "
        "generic name only if the supplied images cannot resolve the type."
    )
