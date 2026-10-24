"""Reuse end-pose segmentation; keep joint-only boundary proposals explicit."""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from atomic_seg.config import SegmentationConfig
from atomic_seg.generation import semantic_action
from atomic_seg.recovery_cot.segmentation import RecoveryProposalConfig, propose_recovery_segments
from atomic_seg.segmentation import _label_arm_segment, segment_episode
from atomic_seg.pretrain.abc_still import absorb_both_still


def segment_native(episode: dict, *, length_policy: dict | None = None) -> dict:
    if episode["profile"] == "realomni_umi":
        from atomic_seg.pretrain.realomni.reference import segment_reference

        return segment_reference(episode, length_policy=length_policy)
    from scipy.spatial.transform import Rotation

    states = np.asarray(episode["states"], dtype=float)
    times = np.asarray(episode["timestamps"], dtype=float)
    n = len(states)
    if n < 2 or not np.isfinite(states).all() or np.any(np.diff(times) <= 0):
        raise ValueError("invalid native trajectory")
    poses = episode["end_effector_poses"]
    endpose = None
    if episode["profile"] == "oxe_bridge":
        quat = Rotation.from_euler("xyz", states[:, 3:6]).as_quat()[:, [3, 0, 1, 2]]
        # Core-only inactive arm. Native exported state/actions remain seven-dimensional.
        inactive = np.zeros((n, 8))
        inactive[:, 3] = 1
        endpose = np.column_stack((states[:, :3], quat, states[:, 6], inactive))
    elif set(poses) == {"left", "right"}:
        parts = []
        for side, grip in [("left", 6), ("right", 13)]:
            matrices = np.asarray(poses[side]).reshape(n, 4, 4)
            if not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-5):
                raise ValueError("invalid homogeneous end-effector pose")
            rotation = matrices[:, :3, :3]
            if not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3):
                raise ValueError("non-orthogonal source end-effector rotation")
            quat = Rotation.from_matrix(rotation).as_quat()[:, [3, 0, 1, 2]]
            # q and -q represent the same orientation; make the trajectory continuous.
            for i in range(1, n):
                if np.dot(quat[i - 1], quat[i]) < 0:
                    quat[i] *= -1
            parts.append(np.column_stack((matrices[:, :3, 3], quat, states[:, grip])))
        endpose = np.concatenate(parts, axis=1)
    if endpose is not None:
        config = SegmentationConfig() if episode["profile"] == "oxe_bridge" else SegmentationConfig(
            gripper_change_threshold=0.12, gripper_event_min_delta=0.12,
            min_segment_seconds=0.3, boundary_merge_seconds=0.25,
            phase_min_seconds=0.3, coordination_min_segment_seconds=0.2,
            still_translation_threshold=0.006, still_path_threshold=0.015,
            still_rotation_threshold=0.05,
        )
        result = segment_episode(endpose, times, config)
        segments = [{"start_frame": s["start_frame"], "end_frame": s["end_frame"],
                     "left_action": semantic_action(s["left_action"], gripper_decrease_means_close=True),
                     "right_action": semantic_action(s["right_action"], gripper_decrease_means_close=True)}
                    for s in result["segments"]]
        method = "existing_state_only_endpose_segmentation"
        evidence = result["boundary_evidence"]
        core_still_refinement = result.get("still_refinement")
    else:
        # This reuses only the numerical proposal helper, not any recovery labels or CoT.
        result = propose_recovery_segments(states, times, episode["state_layout"],
                    config=RecoveryProposalConfig(min_segment_seconds=0.65, max_segment_seconds=4.0,
                                                  smoothing_seconds=0.15, direction_window_seconds=0.3))
        segments = []
        for candidate in result["segments"]:
            start, end = candidate["start_frame"], candidate["end_frame"]
            segment = {"start_frame": start, "end_frame": end}
            for side, offset in [("left", 0), ("right", 7)]:
                delta = states[end, offset + 6] - states[start, offset + 6]
                joint_path = np.abs(np.diff(states[start:end + 1, offset:offset + 6], axis=0)).sum(axis=0)
                gripper_range = np.ptp(states[start:end + 1, offset + 6])
                label = "close_gripper" if delta < -0.12 else "open_gripper" if delta > 0.12 else "move" if max(joint_path, default=0) > 0.04 or gripper_range > 0.08 else "still"
                segment[side + "_action"] = label
            segments.append(segment)
        method = "existing_joint_motion_proposals_with_measured_gripper_labels"
        evidence = [s["start_boundary_evidence"] for s in result["segments"] if s["start_boundary_evidence"]]
        core_still_refinement = None
    # Like the existing semantic cleanup, correct hard gripper contradictions downstream
    # without moving a single boundary. Real aperture feedback may bounce while holding.
    corrections = []
    for segment in segments:
        start, end = segment["start_frame"], segment["end_frame"]
        arms = [("left", 6)] if episode["profile"] == "oxe_bridge" else [("left", 6), ("right", 13)]
        for arm, grip in arms:
            old = segment[arm + "_action"]
            measured = states[end, grip] - states[start, grip]
            commanded = episode["command_endpoints"][end][grip] - states[start, grip]
            wrong = (old == "open_gripper" and measured < 0.12 and commanded < 0.05) or (
                old == "close_gripper" and measured > -0.12 and commanded > -0.05
            )
            if not wrong:
                continue
            if measured >= 0.12:
                new = "open_gripper"
            elif measured <= -0.12:
                new = "close_gripper"
            elif endpose is not None:
                new, _ = _label_arm_segment(endpose, start, end, arm, SegmentationConfig(gripper_change_threshold=10.0))
            else:
                new = "move"
            segment[arm + "_action"] = new
            corrections.append({"start_frame": start, "end_frame": end, "arm": arm,
                                "before": old, "after": new, "measured_aperture_delta": float(measured),
                                "commanded_aperture_delta": float(commanded),
                                "reason": "gripper label contradicted measured change and commanded target"})
    # Reuse the established short same-action principle; do not force a segment count.
    merged = []
    for segment in segments:
        if merged and all(merged[-1][k] == segment[k] for k in ["left_action", "right_action"]) and (
            segment["end_frame"] - segment["start_frame"] + 1 < 8
            or merged[-1]["end_frame"] - merged[-1]["start_frame"] + 1 < 8
        ):
            merged[-1]["end_frame"] = segment["end_frame"]
        else:
            merged.append(dict(segment))
    for i, segment in enumerate(merged):
        segment["segment_index"] = i
        segment["primary_action_verb"] = (
            f"arm:{segment['left_action']}" if episode["profile"] == "oxe_bridge"
            else f"left:{segment['left_action']}, right:{segment['right_action']}"
        )
    length_audit = None
    original_segments = merged
    both_still_audit = None
    if episode["profile"] == "abc_yam":
        merged, both_still_audit = absorb_both_still(merged)
    if length_policy is not None:
        from atomic_seg.pretrain.length_policy import LengthPolicy, coarsen_segments, length_statistics

        merged, length_audit = coarsen_segments(episode, merged, LengthPolicy(**length_policy))
        length_audit["after_still_cleanup"] = length_audit["before"]
        length_audit["before"] = length_statistics(original_segments)
        length_audit["both_still_cleanup"] = both_still_audit
    return {"method": method, "segments": merged, "boundary_evidence": evidence,
            "length_audit": length_audit, "original_segments": original_segments,
            "both_still_cleanup": both_still_audit, "core_still_refinement": core_still_refinement,
            "config": asdict(config) if endpose is not None else result["config"],
            "label_corrections": corrections,
            "source_annotation_policy": (
                "retain nearest existing motion boundary; offsets recorded in length_audit"
                if length_policy is not None else "context only, not forced boundaries"
            ),
            "boundary_inputs": "measured end-effector states" if endpose is not None else "measured native joint/gripper states"}


def target_for(episode: dict, segment: dict) -> dict:
    start, end = segment["start_frame"], segment["end_frame"]
    endpoint = np.asarray(episode["command_endpoints"][end])
    current = np.asarray(episode["states"][start])
    return {"guide_action": np.round(endpoint - current, 8).tolist(),
            "primary_action_verb": segment["primary_action_verb"],
            "sub_task": segment["sub_task"], "num_chunks": math.ceil((end - start + 1) / 8)}
