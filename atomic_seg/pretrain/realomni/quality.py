"""Whole-file quality decisions. Never salvage a time window from a bad episode."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .config import Settings
from .reference import reference_episode


@dataclass
class Stream:
    timestamps: list = field(default_factory=list)
    values: list = field(default_factory=list)
    sequences: list = field(default_factory=list)


def issue(code, **details):
    return {"code": code, **details}


def check_stream(name, stream, hz, settings):
    t = np.asarray(stream.timestamps, dtype=np.int64)
    errors = []
    report = {"topic": name, "count": len(t), "expected_hz": hz}
    if len(t) < 2:
        return report, [issue("missing_stream_or_samples", topic=name, count=len(t))]
    differences = np.diff(t)
    bad = np.flatnonzero((differences <= 0) | (differences > settings.maximum_gap_periods * 1e9 / hz))
    report.update(first_timestamp_ns=int(t[0]), last_timestamp_ns=int(t[-1]),
                  maximum_gap_ns=int(differences.max()), gap_count=len(bad))
    if len(bad):
        errors.append(issue("timestamp_gap_or_disorder", topic=name, count=len(bad),
                            first_indices=bad[:20].tolist()))
    # VIO's sequence_num tracks reconstruction updates: real files repeat it
    # while publishing distinct 30 Hz poses. It is not a pose sample counter.
    sequence_is_sample_counter = "/sensor/" in name
    report["sequence_policy"] = "sample_counter" if sequence_is_sample_counter else "not_a_sample_counter"
    if sequence_is_sample_counter and stream.sequences and any(stream.sequences):
        # Header sequence is uint32 and may legitimately wrap once.
        sequence = np.asarray(stream.sequences, dtype=np.int64)
        bad_sequence = np.flatnonzero((np.diff(sequence) % 2**32) != 1)
        if len(bad_sequence):
            errors.append(issue("sequence_gap_or_disorder", topic=name, count=len(bad_sequence),
                                first_indices=bad_sequence[:20].tolist()))
    if stream.values and not np.isfinite(np.asarray(stream.values)).all():
        errors.append(issue("nonfinite_measurement", topic=name))
    if name.endswith("eef_pose") and stream.values:
        values = np.asarray(stream.values, dtype=float)
        if values.shape != (len(t), 7) or not np.allclose(np.linalg.norm(values[:, 3:], axis=1), 1, atol=1e-3, rtol=0):
            errors.append(issue("invalid_source_pose", topic=name))
    return report, errors


BASE_POLICY = {"status": "not_evaluated", "filter_enabled": False,
               "reason": "handheld human collection; chassis filtering disabled by user"}


def assess_streams(streams, settings):
    if settings.quality_policy == "trim_edges_1s":
        from .trim import assess_trimmed_streams

        return assess_trimmed_streams(streams, settings)
    errors, reports = [], {}
    rates = {}
    for robot in ("robot0", "robot1"):
        rates.update({f"/{robot}/vio/eef_pose": settings.pose_hz,
                      f"/{robot}/sensor/camera0/compressed": settings.camera_hz,
                      f"/{robot}/sensor/magnetic_encoder": settings.gripper_hz,
                      f"/{robot}/sensor/imu": settings.imu_hz})
    for topic, hz in rates.items():
        report, failures = check_stream(topic, streams.get(topic, Stream()), hz, settings)
        reports[topic] = report
        errors.extend(failures)
    available = [reports[t] for t in rates if reports[t]["count"] >= 2]
    if not available:
        return {"streams": reports, "base": BASE_POLICY, "reasons": errors, "whole_episode": True}, None
    first = min(r["first_timestamp_ns"] for r in available)
    last = max(r["last_timestamp_ns"] for r in available)
    for topic, hz in rates.items():
        report = reports[topic]
        if report["count"] < 2:
            continue
        margin = settings.maximum_boundary_skew_periods * 1e9 / hz
        if report["first_timestamp_ns"] - first > margin or last - report["last_timestamp_ns"] > margin:
            errors.append(issue("missing_episode_prefix_or_suffix", topic=topic,
                                prefix_ns=report["first_timestamp_ns"] - first,
                                suffix_ns=last - report["last_timestamp_ns"]))
    audit = {"streams": reports, "base": BASE_POLICY, "reasons": errors,
             "whole_episode": True, "trimmed_observations": 0, "interpolated_observations": 0}
    if errors:
        return audit, None
    clock = np.asarray(streams["/robot0/vio/eef_pose"].timestamps, dtype=np.int64)
    if not 3 <= len(clock) <= settings.maximum_episode_rows:
        errors.append(issue("episode_row_limit", count=len(clock)))
        return audit, None
    arrays, alignment, camera_indices = [], {}, {}
    for robot, side in (("robot0", "left"), ("robot1", "right")):
        parts = []
        for suffix in ("vio/eef_pose", "sensor/magnetic_encoder", "sensor/camera0/compressed"):
            topic = f"/{robot}/{suffix}"
            stream = streams[topic]
            stamps = np.asarray(stream.timestamps, dtype=np.int64)
            indices = np.searchsorted(stamps, clock, side="right") - 1
            if np.any(indices < 0):
                errors.append(issue("no_causal_observation", topic=topic))
                continue
            ages = clock - stamps[indices]
            if np.any(ages > settings.maximum_alignment_age_periods * 1e9 / rates[topic]):
                errors.append(issue("stale_observation", topic=topic, max_age_ns=int(ages.max())))
                continue
            alignment[topic] = {"method": "latest_at_or_before", "source_indices": indices.tolist(),
                                "source_timestamps_ns": stamps[indices].tolist(), "max_abs_skew_ms": float(ages.max()) / 1e6}
            if suffix.endswith("compressed"):
                camera_indices[side + "_wrist"] = indices.tolist()
            else:
                values = np.asarray(stream.values)[indices]
                parts.append(values if suffix.endswith("eef_pose") else values.reshape(-1, 1) / settings.gripper_scale_m)
        if len(parts) == 2:
            arrays.extend(parts)
    if errors:
        return audit, None
    try:
        episode = reference_episode(np.concatenate(arrays, axis=1), clock)
    except ValueError as error:
        errors.append(issue("invalid_end_effector_state", detail=str(error)))
        return audit, None
    episode["alignment"] = {"source_rows": len(clock), "reference_action_rows": len(clock) - 1,
                            "dropped_observations": 0, "streams": alignment}
    episode["camera_source_indices"] = camera_indices
    return audit, episode


def pose_values(pose):
    return [pose.position.x, pose.position.y, pose.position.z,
            pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]


def check_video(path, expected, *, retained_interval=None):
    """Decode every frame without JPEG output; headers alone cannot prove completeness."""
    import av
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    contexts, codecs, decoded = {}, {}, {topic: [] for topic in expected}
    edge_errors = []

    def retained(stamp):
        return retained_interval is None or retained_interval[0] <= stamp <= retained_interval[1]

    def accept(topic, frames):
        for frame in frames:
            if frame.pts is None:
                raise ValueError(f"corrupt or untimestamped frame: {topic}")
            stamp = int(frame.pts)
            if not retained(stamp):
                continue
            if frame.is_corrupt:
                raise ValueError(f"corrupt retained frame: {topic}, {stamp}")
            decoded[topic].append(stamp)

    with Path(path).open("rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for _, channel, _, message in reader.iter_decoded_messages(topics=list(expected), log_time_order=False):
            topic, codec = channel.topic, message.format
            if topic not in contexts:
                if codec not in {"h264", "h265", "hevc"}:
                    raise ValueError(f"unsupported video codec {codec}: {topic}")
                contexts[topic] = av.CodecContext.create("hevc" if codec == "h265" else codec, "r")
                contexts[topic].thread_count = 2
                contexts[topic].options = {"err_detect": "explode"}
                codecs[topic] = codec
            if codec != codecs[topic]:
                raise ValueError(f"changed codec: {topic}")
            stamp = int(message.header.timestamp)
            if not message.data:
                if retained(stamp):
                    raise ValueError(f"empty retained packet: {topic}")
                edge_errors.append({"topic": topic, "timestamp_ns": stamp, "reason": "empty cropped packet"})
                continue
            packet = av.Packet(bytes(message.data))
            packet.pts = int(message.header.timestamp)
            packet.time_base = Fraction(1, 1_000_000_000)
            try:
                frames = contexts[topic].decode(packet)
            except av.FFmpegError as error:
                if retained(stamp):
                    raise
                edge_errors.append({"topic": topic, "timestamp_ns": stamp, "reason": str(error)})
                continue
            accept(topic, frames)
    for topic, context in contexts.items():
        accept(topic, context.decode(None))
    report = {}
    for topic, stamps in expected.items():
        wanted = [t for t in stamps if retained(t)]
        if sorted(decoded[topic]) != sorted(wanted):
            raise ValueError(f"decoded video timestamps/count differ from complete stream: {topic}")
        report[topic] = {"decoded_frames": len(decoded[topic]), "codec": codecs[topic],
                         "scope": "retained_interval" if retained_interval else "whole_source",
                         "excluded_edge_decode_errors": [e for e in edge_errors if e["topic"] == topic]}
    return report


def _inspect_episode(path, settings, checkpoint):
    import av
    from mcap.reader import make_reader, NonSeekingReader
    from mcap_protobuf.decoder import DecoderFactory

    streams, counts, errors, calibration = {}, Counter(), [], {}
    checkpoint("reading_source", {})
    # Exceptions from storage/decoding are retryable errors in the runner. They
    # must not become permanent quality rejections merely because I/O failed.
    with Path(path).open("rb") as f:
        magic = b"\x89MCAP0\r\n"
        f.seek(0, 2)
        size = f.tell()
        f.seek(0)
        prefix = f.read(len(magic))
        f.seek(max(0, size - len(magic)))
        suffix = f.read(len(magic))
        if size < 2 * len(magic) or prefix != magic or suffix != magic:
            return {"whole_episode": True, "base": BASE_POLICY,
                    "reasons": [issue("corrupt_or_truncated_mcap", detail="missing file header or footer magic")],
                    "video": {"status": "not_checked"}}, None
        f.seek(0)
        reader = make_reader(f, decoder_factories=[DecoderFactory()], validate_crcs=True)
        summary = reader.get_summary()
        if summary is None or summary.statistics is None:
            return {"whole_episode": True, "base": BASE_POLICY, "reasons": [issue("missing_mcap_summary")], "video": {"status": "not_checked"}}, None
        channels = summary.channels
        expected = summary.statistics.channel_message_counts
        f.seek(0)
        # Read every physical record, including data-section CRC, rather than
        # trusting chunk indexes to enumerate the whole file.
        reader = NonSeekingReader(f, decoder_factories=[DecoderFactory()], validate_crcs=True)
        for _, channel, record, message in reader.iter_decoded_messages(log_time_order=False):
            counts[channel.id] += 1
            topic = channel.topic
            header = getattr(message, "header", None)
            stamp = int(getattr(header, "timestamp", 0))
            if topic.endswith("camera_info"):
                calibration[topic] = bool(message.width and message.height and len(message.K) == 9)
                continue
            values = None
            if topic.endswith("/vio/eef_pose"):
                values = pose_values(message.pose)
            elif topic.endswith("/sensor/magnetic_encoder"):
                values = [message.value]
            elif topic.endswith("/sensor/imu"):
                values = [message.angular_velocity.x, message.angular_velocity.y, message.angular_velocity.z,
                          message.linear_acceleration.x, message.linear_acceleration.y, message.linear_acceleration.z]
            elif topic.endswith("/sensor/camera0/compressed"):
                if not message.data:
                    errors.append(issue("empty_video_packet", topic=topic, timestamp_ns=stamp))
            else:
                # Human collection: no chassis inference or motion rejection.
                continue
            if not stamp:
                errors.append(issue("missing_source_timestamp", topic=topic))
                stamp = int(record.log_time)  # Diagnostic only: error excludes this whole episode.
            stream = streams.setdefault(topic, Stream())
            stream.timestamps.append(stamp)
            stream.sequences.append(int(getattr(header, "sequence_num", 0)))
            if values is not None:
                stream.values.append(values)
            if len(stream.timestamps) > settings.maximum_episode_rows * 10:
                raise ValueError(f"source stream exceeds bounded episode memory: {topic}")
        for channel_id, channel in channels.items():
            if counts[channel_id] != expected.get(channel_id, 0):
                errors.append(issue("mcap_message_count_mismatch", topic=channel.topic,
                                    expected=expected.get(channel_id, 0), actual=counts[channel_id]))
    for robot in ("robot0", "robot1"):
        topic = f"/{robot}/sensor/camera0/camera_info"
        if not calibration.get(topic):
            errors.append(issue("missing_camera_calibration", topic=topic))
    audit, episode = assess_streams(streams, settings)
    if settings.quality_policy == "trim_edges_1s":
        failures = []
        for error in errors:
            optional = error.get("topic", "").endswith(("/imu", "/camera_info"))
            crop = audit.get("crop")
            outside = (error["code"] == "empty_video_packet" and crop is not None and
                       not crop["common_start_ns"] <= error["timestamp_ns"] <= crop["common_end_ns"])
            (audit["diagnostics"] if optional or outside else failures).append(error)
        errors = failures
    audit["reasons"] = errors + audit["reasons"]
    audit["video"] = {"status": "not_checked", "reason": "earlier quality gate failed"}
    checkpoint("checked_numeric", audit)
    if audit["reasons"]:
        return audit, None
    checkpoint("checking_video", audit)
    try:
        crop = audit.get("crop")
        interval = (crop["common_start_ns"], crop["common_end_ns"]) if crop else None
        report = check_video(path, {topic: stream.timestamps for topic, stream in streams.items() if topic.endswith("compressed")},
                             retained_interval=interval)
    except (ValueError, av.FFmpegError) as error:
        audit["reasons"].append(issue("incomplete_or_corrupt_video", detail=str(error)))
        audit["video"] = {"status": "failed", "detail": str(error)}
        return audit, None
    audit["video"] = {"status": "passed", "streams": report}
    audit["status"] = "passed"
    episode["native_aux"] = {topic: {"timestamps_ns": s.timestamps, "values": s.values,
                                     "sequences": s.sequences} for topic, s in streams.items()}
    return audit, episode


def inspect_episode(path, settings=None, *, checkpoint=None):
    import struct
    from google.protobuf.message import DecodeError
    from mcap.exceptions import EndOfFile, InvalidMagic
    from mcap.stream_reader import CRCValidationError

    checkpoint = checkpoint or (lambda stage, details: None)
    try:
        return _inspect_episode(path, settings or Settings(), checkpoint)
    except (EndOfFile, InvalidMagic, CRCValidationError, DecodeError, struct.error) as error:
        # Corrupt source data is a durable rejection; I/O and unsupported
        # schemas/dependencies propagate to the runner as retryable errors.
        return {"whole_episode": True, "base": BASE_POLICY,
                "reasons": [issue("corrupt_or_truncated_mcap", error_type=type(error).__name__, detail=str(error))],
                "video": {"status": "not_checked"}}, None
