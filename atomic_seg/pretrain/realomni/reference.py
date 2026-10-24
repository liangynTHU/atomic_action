"""Measured successor targets and the shared 16-component planner contract."""

from __future__ import annotations

import math

import numpy as np

from atomic_seg.segmentation import STATE_LAYOUT, segment_episode
from atomic_seg.config import SegmentationConfig
from atomic_seg.generation import semantic_action
from atomic_seg.pretrain.abc_still import absorb_both_still
from atomic_seg.pretrain.length_policy import LengthPolicy, coarsen_segments


ACTION_CONTRACT = {
    "state_layout": list(STATE_LAYOUT),
    "dimension": 16,
    "reference_action": "action[t] = measured_state[t + 1]",
    "guide_action": "action[end] - state[start] = measured_state[end + 1] - measured_state[start]",
    "quaternion": "wxyz; sign-continuous unit states; guide uses component differences, not relative rotation",
    "gripper": "measured aperture_m / 0.103; 0=closed, 1=open; delta=0 means unchanged",
    "num_chunks": "ceil(valid_action_rows / 8)",
    "terminal_observation": "retained; no synthetic action after the final measurement",
    "provenance": "reference targets derived from measurements, not native controller commands",
}


def reference_episode(measured_states, timestamps_ns, *, task="", episode_id="", source=None):
    measured = np.asarray(measured_states, dtype=np.float64).copy()
    stamps = np.asarray(timestamps_ns, dtype=np.int64)
    if measured.ndim != 2 or measured.shape[1] != 16 or len(measured) < 3:
        raise ValueError("reference trajectory requires at least three 16D observations")
    if not np.isfinite(measured).all() or stamps.shape != (len(measured),) or np.any(np.diff(stamps) <= 0):
        raise ValueError("invalid measured trajectory or clock")
    for offset in (3, 11):
        quaternion = measured[:, offset:offset + 4]
        norms = np.linalg.norm(quaternion, axis=1)
        if not np.allclose(norms, 1, atol=1e-3, rtol=0):
            raise ValueError("invalid measured quaternion")
        quaternion /= norms[:, None]
        for i in range(1, len(quaternion)):
            if np.dot(quaternion[i - 1], quaternion[i]) < 0:
                quaternion[i] *= -1
    return {
        "id": episode_id, "dataset": "RealOmni Open", "subset": "umi", "profile": "realomni_umi",
        "task": task, "source": source or {}, "state_layout": list(STATE_LAYOUT),
        "action_layout": list(STATE_LAYOUT), "action_contract": ACTION_CONTRACT,
        "measured_states": measured.tolist(), "measured_timestamps_ns": stamps.tolist(),
        "states": measured[:-1].tolist(), "actions": measured[1:].tolist(),
        "command_endpoints": measured[1:].tolist(), "timestamps_ns": stamps[:-1].tolist(),
        "timestamps": ((stamps[:-1] - stamps[0]) / 1e9).tolist(),
        "time_basis": "source header nanoseconds; left measured pose clock; causal observations",
        "end_effector_poses": {}, "source_annotations": [], "native_aux": {},
    }


def segment_reference(episode, *, length_policy=None):
    """Segment all observations, then partition exactly N-1 real transitions."""
    measured = np.asarray(episode["measured_states"])
    n = len(measured) - 1
    if (measured.shape != (n + 1, 16) or n < 2
            or not np.array_equal(episode["states"], measured[:-1])
            or not np.array_equal(episode["actions"], measured[1:])
            or not np.array_equal(episode["command_endpoints"], measured[1:])):
        raise ValueError("reference actions must be adjacent measured 16D states")
    stamps = np.asarray(episode["measured_timestamps_ns"], dtype=np.int64)
    crop = episode.get("crop")
    config = SegmentationConfig(
        gripper_change_threshold=0.12, gripper_event_min_delta=0.12,
        min_segment_seconds=0.3, boundary_merge_seconds=0.25,
        phase_min_seconds=0.3, coordination_min_segment_seconds=0.2,
        still_translation_threshold=0.006, still_path_threshold=0.015,
        still_rotation_threshold=0.05,
    )
    result = segment_episode(measured, (stamps - stamps[0]) / 1e9, config)
    primitives = []
    for s in result["segments"]:
        if s["start_frame"] >= n:
            continue  # The terminal observation does not have an outgoing action.
        primitives.append({
            "segment_index": len(primitives), "start_frame": s["start_frame"],
            "end_frame": min(s["end_frame"], n - 1),
            **{arm + "_action": semantic_action(s[arm + "_action"], gripper_decrease_means_close=True)
               for arm in ("left", "right")},
        })
    segments, still_audit = absorb_both_still(primitives)
    length_audit = None
    if length_policy is not None:
        segments, length_audit = coarsen_segments(episode, segments, LengthPolicy(**length_policy))
    cursor = 0
    for s in segments:
        start, end = s["start_frame"], s["end_frame"]
        if start != cursor or not start <= end < n:
            raise ValueError("reference segmentation has a gap or overlap")
        s.update(valid_action_rows=end - start + 1, target_observation_frame=end + 1,
                 guide_action=np.round(measured[end + 1] - measured[start], 8).tolist(),
                 num_chunks=math.ceil((end - start + 1) / 8))
        if crop:
            origin = crop["original_start_ns"]
            s.update(source_start_frame=crop["source_left_start_frame"] + start,
                     source_target_frame=crop["source_left_start_frame"] + end + 1,
                     start_time_seconds=float(stamps[start] - origin) / 1e9,
                     target_time_seconds=float(stamps[end + 1] - origin) / 1e9)
        cursor = end + 1
    if cursor != n:
        raise ValueError("reference segmentation did not cover every measured transition")
    return {
        "method": "shared_endpose_segmentation_on_all_measured_observations",
        "action_contract": ACTION_CONTRACT, "segments": segments, "original_segments": primitives,
        "boundary_evidence": result["boundary_evidence"], "core_still_refinement": result.get("still_refinement"),
        "both_still_cleanup": still_audit, "length_audit": length_audit,
        "measured_observations": n + 1, "reference_action_rows": n,
        "full_transition_coverage": True, "terminal_observation_retained": True,
        "crop": crop, "coverage_scope": "retained_interval" if crop else "whole_source",
    }
