"""One-second edge cropping; no internal-gap stitching or IMU rejection."""

from __future__ import annotations

import numpy as np

from .reference import reference_episode


def assess_trimmed_streams(streams, settings):
    from .quality import BASE_POLICY, Stream, check_stream, issue

    errors, diagnostics, full_reports = [], [], {}
    rates = {f"/robot{robot}/{suffix}": hz for robot in (0, 1) for suffix, hz in (
        ("vio/eef_pose", settings.pose_hz), ("sensor/magnetic_encoder", settings.gripper_hz),
        ("sensor/camera0/compressed", settings.camera_hz))}
    for robot in (0, 1):
        topic = f"/robot{robot}/sensor/imu"
        report, failures = check_stream(topic, streams.get(topic, Stream()), settings.imu_hz, settings)
        report["required_for_segmentation"] = False
        full_reports[topic] = report
        diagnostics.extend(failures)
    audit = {"policy": "trim_edges_1s", "base": BASE_POLICY, "reasons": errors,
             "diagnostics": diagnostics, "streams": full_reports,
             "whole_episode": False, "internal_gap_stitching": False,
             "interpolated_observations": 0, "trimmed_observations": 0,
             "imu_policy": "diagnostic_only; not used for cropping, segmentation, action or guide",
             "calibration_policy": "diagnostic_only; raw images, no undistortion or projection"}
    clocks = {}
    for topic, hz in rates.items():
        stream = streams.get(topic, Stream())
        report, _ = check_stream(topic, stream, hz, settings)
        report["required_for_segmentation"] = True
        full_reports[topic] = report
        clock = np.asarray(stream.timestamps, dtype=np.int64)
        clocks[topic] = clock
        if len(clock) < 2:
            errors.append(issue("missing_stream_or_samples", topic=topic, count=len(clock)))
        elif np.any(np.diff(clock) <= 0):
            errors.append(issue("source_clock_disorder", topic=topic))
    if errors:
        return audit, None
    original_start = min(int(t[0]) for t in clocks.values())
    original_end = max(int(t[-1]) for t in clocks.values())
    common_start = max(int(t[0]) for t in clocks.values())
    common_end = min(int(t[-1]) for t in clocks.values())
    if common_end <= common_start:
        errors.append(issue("no_common_required_stream_interval"))
        return audit, None
    # All channels supply at least one in-window observation before the first
    # reference timestamp. A phase shift never permits use of a future frame.
    first_indices = {topic: int(np.searchsorted(t, common_start)) for topic, t in clocks.items()}
    if any(i >= len(clocks[topic]) for topic, i in first_indices.items()):
        errors.append(issue("no_common_required_stream_interval"))
        return audit, None
    causal_start = max(int(clocks[topic][i]) for topic, i in first_indices.items())
    left_clock = clocks["/robot0/vio/eef_pose"]
    left_start = int(np.searchsorted(left_clock, causal_start))
    left_stop = int(np.searchsorted(left_clock, common_end, side="right"))
    clock = left_clock[left_start:left_stop]
    if len(clock) < 3 or len(clock) > settings.maximum_episode_rows:
        errors.append(issue("insufficient_retained_observations", count=len(clock)))
        return audit, None
    prefix_s = (int(clock[0]) - original_start) / 1e9
    suffix_s = (original_end - int(clock[-1])) / 1e9
    crop = {"original_start_ns": original_start, "original_end_ns": original_end,
            "common_start_ns": common_start, "common_end_ns": common_end,
            "retained_start_ns": int(clock[0]), "retained_end_ns": int(clock[-1]),
            "prefix_seconds": prefix_s, "suffix_seconds": suffix_s,
            "maximum_seconds_per_edge": settings.maximum_edge_trim_seconds,
            "source_left_start_frame": left_start, "source_left_end_frame": left_stop - 1,
            "source_left_observations": len(left_clock), "retained_observations": len(clock),
            "trimmed_left_prefix_rows": left_start, "trimmed_left_suffix_rows": len(left_clock) - left_stop,
            "per_stream": {},
            "alignment_context": "causal samples inside common interval may precede first reference timestamp by a stream phase"}
    audit["crop"] = crop
    audit["trimmed_observations"] = len(left_clock) - len(clock)
    if prefix_s > settings.maximum_edge_trim_seconds or suffix_s > settings.maximum_edge_trim_seconds:
        errors.append(issue("edge_trim_exceeds_one_second", prefix_seconds=prefix_s, suffix_seconds=suffix_s))
        return audit, None
    arrays, alignment, camera_indices = [], {}, {}
    retained_reports = {}
    audit["retained_streams"] = retained_reports
    for robot, side in ((0, "left"), (1, "right")):
        for suffix in ("vio/eef_pose", "sensor/magnetic_encoder", "sensor/camera0/compressed"):
            topic = f"/robot{robot}/{suffix}"
            t, stream = clocks[topic], streams[topic]
            lo = first_indices[topic]
            hi = int(np.searchsorted(t, common_end, side="right"))
            sliced = Stream(t[lo:hi].tolist(), stream.values[lo:hi] if stream.values else [],
                            stream.sequences[lo:hi] if stream.sequences else [])
            report, failures = check_stream(topic, sliced, rates[topic], settings)
            report["source_start_index"] = lo
            retained_reports[topic] = report
            errors.extend(failures)
            crop["per_stream"][topic] = {"source_rows": len(t), "retained_rows": hi - lo,
                                        "prefix_rows": lo, "suffix_rows": len(t) - hi}
            indices = np.searchsorted(t, clock, side="right") - 1
            if np.any(indices < lo) or np.any(indices >= hi):
                errors.append(issue("no_in_window_causal_observation", topic=topic))
                continue
            ages = clock - t[indices]
            if np.any(ages > settings.maximum_alignment_age_periods * 1e9 / rates[topic]):
                errors.append(issue("stale_observation", topic=topic, max_age_ns=int(ages.max())))
            alignment[topic] = {"method": "latest_at_or_before", "source_indices": indices.tolist(),
                                "source_timestamps_ns": t[indices].tolist(),
                                "max_abs_skew_ms": float(ages.max()) / 1e6}
            if suffix.endswith("compressed"):
                camera_indices[side + "_wrist"] = indices.tolist()
            else:
                values = np.asarray(stream.values)[indices]
                arrays.append(values if suffix.endswith("eef_pose") else values.reshape(-1, 1) / settings.gripper_scale_m)
    if errors:
        return audit, None
    try:
        episode = reference_episode(np.concatenate(arrays, axis=1), clock)
    except ValueError as error:
        errors.append(issue("invalid_end_effector_state", detail=str(error)))
        return audit, None
    episode.update(crop=crop, camera_source_indices=camera_indices,
                   alignment={"source_rows": len(left_clock), "retained_observations": len(clock),
                              "reference_action_rows": len(clock) - 1,
                              "dropped_observations": len(left_clock) - len(clock), "streams": alignment})
    return audit, episode
