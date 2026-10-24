"""Read native trajectories without assuming that heterogeneous actions are 16-D."""

from __future__ import annotations

import io
import struct
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image

from atomic_seg.common.artifacts import sha256_file


def align_indices(source_times, target_times, *, causal: bool):
    source = np.asarray(source_times, dtype=np.int64)
    target = np.asarray(target_times, dtype=np.int64)
    if len(source) == 0 or np.any(np.diff(source) < 0):
        raise ValueError("source timestamps must be nonempty and sorted")
    right = np.searchsorted(source, target, side="right")
    if causal:
        result = right - 1
        if np.any(result < 0):
            raise ValueError("no past observation for an action timestamp")
    else:
        lo = np.clip(right - 1, 0, len(source) - 1)
        hi = np.clip(right, 0, len(source) - 1)
        result = np.where(abs(source[lo] - target) <= abs(source[hi] - target), lo, hi)
    return result


def tfrecord_examples(path):
    # Native Example decoding avoids importing TensorFlow or loading a whole shard.
    import crc32c
    from tfrecord.example_pb2 import Example

    def crc(data):
        n = crc32c.crc32c(data)
        return (((n >> 15) | (n << 17)) + 0xA282EAD8) & 0xFFFFFFFF

    with Path(path).open("rb") as stream:
        index = 0
        while header := stream.read(12):
            if len(header) != 12:
                raise ValueError("truncated TFRecord header")
            length, checksum = struct.unpack("<QI", header)
            if crc(header[:8]) != checksum or length > 2**31:
                raise ValueError("invalid TFRecord length/CRC")
            record = stream.read(length)
            footer = stream.read(4)
            if len(record) != length or len(footer) != 4 or crc(record) != struct.unpack("<I", footer)[0]:
                raise ValueError("truncated/corrupt TFRecord")
            example = Example.FromString(record)
            values = {
                key: list(getattr(value, value.WhichOneof("kind")).value)
                for key, value in example.features.feature.items()
            }
            import hashlib

            yield index, values, hashlib.sha256(record).hexdigest()
            index += 1


def load_bridge(root: Path, choice: dict, cache: Path) -> dict:
    version = root / "bridge" / "0.1.0"
    shard = version / choice["shard"]
    for record_index, values, record_hash in tfrecord_examples(shard):
        if record_index == choice["record_index"]:
            break
    else:
        raise ValueError("missing requested TFRecord episode")
    def get(name):
        return values["steps/" + name]
    n = len(get("is_first"))
    if get("is_first") != [1] + [0] * (n - 1) or get("is_last") != [0] * (n - 1) + [1]:
        raise ValueError("episode is not complete")
    instructions = {s.decode("utf-8").strip() for s in get("observation/natural_language_instruction") if s}
    if len(instructions) != 1:
        raise ValueError("Bridge episode must contain one nonempty instruction")
    states = np.asarray(get("observation/state")).reshape(n, 7)
    translations = np.asarray(get("action/world_vector")).reshape(n, 3)
    rotations = np.asarray(get("action/rotation_delta")).reshape(n, 3)
    apertures = np.asarray(get("action/open_gripper"))
    actions = np.column_stack((translations, rotations, apertures))
    endpoints = np.column_stack((states[:, :6] + actions[:, :6], apertures))
    images = []
    for i, data in enumerate(get("observation/image")):
        path = cache / "image" / f"{i:06d}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(io.BytesIO(data)).convert("RGB").save(path, quality=92)
        images.append(str(path.resolve()))
    if len(images) != n:
        raise ValueError("image/state count mismatch")
    # This RLDS export has no timestamps. Keep frame indices and declare the time assumption.
    fps = float(choice.get("nominal_fps", 5))
    return {
        "dataset": "OXE", "subset": "bridge", "profile": "oxe_bridge",
        "task": next(iter(instructions)), "states": states.tolist(),
        "actions": actions.tolist(), "command_endpoints": endpoints.tolist(),
        "timestamps": (np.arange(n) / fps).tolist(),
        "time_basis": f"nominal {fps:g} Hz; source RLDS contains no timestamps",
        "state_layout": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        "action_layout": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "open_gripper"],
        "camera_paths": {"image": images}, "camera_source_indices": {"image": list(range(n))},
        "native_aux": {k: get(k) for k in ["reward", "is_first", "is_last", "is_terminal", "action/terminate_episode"]},
        "source_annotations": [], "end_effector_poses": {},
        "source": {"root": str(root), "file": str(shard), "record_index": record_index,
                   "record_sha256": record_hash, "split": choice["shard"].split("-")[1].split(".")[0]},
        "alignment": {"method": "RLDS step identity", "source_rows": n, "dropped_action_rows": 0},
    }


def load_abc(root: Path, choice: dict, cache: Path) -> dict:
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    directory = root / choice["episode"]
    path = directory / "episode.mcap"
    streams, cameras, contexts, frame_paths = defaultdict(list), defaultdict(list), {}, defaultdict(list)
    task, poses, codecs = "", defaultdict(list), {}

    def save_frame(topic, frame):
        if frame.pts is None:
            raise ValueError("video decoder did not preserve timestamps")
        stamp = int(frame.pts)
        name = topic.strip("/").replace("-camera", "").replace("-", "_")
        output = cache / name / f"{stamp}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        image = frame.to_image()
        image.thumbnail((480, 360))
        image.save(output, quality=86)
        cameras[name].append(stamp)
        frame_paths[name].append(str(output.resolve()))

    with path.open("rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for _, channel, message, decoded in reader.iter_decoded_messages():
            topic = channel.topic
            stamp = int(message.log_time)
            if topic == "/instruction":
                task = str(decoded.data)
            elif topic.endswith("-camera"):
                codec = str(decoded.format)
                if topic not in contexts:
                    contexts[topic] = av.CodecContext.create("hevc" if codec == "h265" else codec, "r")
                    codecs[topic] = codec
                if codecs[topic] != codec:
                    raise ValueError("video codec changes within a stream")
                packet = av.Packet(bytes(decoded.data))
                packet.pts = stamp
                packet.time_base = Fraction(1, 1_000_000_000)
                for frame in contexts[topic].decode(packet):
                    save_frame(topic, frame)
            elif topic.endswith(("-state", "-action")):
                streams[topic].append((stamp, list(decoded.position)))
                if topic.endswith("-arm-state"):
                    poses[topic].append(list(decoded.pose))
        summary = reader.get_summary()
        counts = {c.topic: summary.statistics.channel_message_counts[c.id] for c in summary.channels.values()}
    for topic, context in contexts.items():
        for frame in context.decode(None):
            save_frame(topic, frame)
        name = topic.strip("/").replace("-camera", "").replace("-", "_")
        order = np.argsort(cameras[name])
        cameras[name] = np.asarray(cameras[name], dtype=np.int64)[order].tolist()
        frame_paths[name] = [frame_paths[name][i] for i in order]
        if len(cameras[name]) != counts[topic]:
            raise ValueError(f"decoded frame count mismatch: {topic}")
    if not task or not cameras:
        raise ValueError("ABC episode is missing instruction or cameras")
    required = [f"/{side}-{part}-{kind}" for kind in ["state", "action"] for side in ["left", "right"] for part in ["arm", "ee"]]
    if any(not streams[t] for t in required):
        raise ValueError("missing ABC robot stream")
    clock = np.asarray([t for t, _ in streams["/left-arm-action"]], dtype=np.int64)
    if np.any(np.diff(clock) <= 0):
        raise ValueError("action clock must be strictly increasing")
    arrays, alignment, observed_poses = {}, {}, {}
    for topic in required:
        stamps = np.asarray([t for t, _ in streams[topic]], dtype=np.int64)
        indices = align_indices(stamps, clock, causal=topic.endswith("-state"))
        if topic.endswith("-action") and (len(stamps) != len(clock) or len(np.unique(indices)) != len(stamps)):
            raise ValueError(f"cannot preserve all command rows on the left-arm clock: {topic}")
        delay = clock - stamps[indices]
        # Do not silently associate missing/remote measurements with actions.
        if np.max(abs(delay)) > 250_000_000:
            raise ValueError(f"robot stream alignment exceeds 250 ms: {topic}")
        arrays[topic] = np.asarray([p for _, p in streams[topic]])[indices]
        expected = 6 if "-arm-" in topic else 1
        if arrays[topic].shape != (len(clock), expected):
            raise ValueError(f"unexpected dimensions: {topic}")
        alignment[topic] = {"method": "latest_at_or_before" if topic.endswith("-state") else "nearest_command",
                            "max_abs_skew_ms": float(np.max(abs(delay))) / 1e6,
                            "source_indices": indices.tolist(), "source_timestamps_ns": stamps[indices].tolist()}
        if topic in poses and all(len(p) == 16 for p in poses[topic]):
            observed_poses[topic.split("-")[0].strip("/")] = np.asarray(poses[topic])[indices].tolist()
    image_rows, image_indices = {}, {}
    for name, stamps in cameras.items():
        indices = align_indices(stamps, clock, causal=True)
        skew = clock - np.asarray(stamps, dtype=np.int64)[indices]
        if np.max(skew) > 250_000_000:
            raise ValueError(f"camera alignment exceeds 250 ms: {name}")
        image_rows[name] = [frame_paths[name][i] for i in indices]
        image_indices[name] = indices.tolist()
        alignment[name] = {"method": "latest_at_or_before", "max_abs_skew_ms": float(max(skew)) / 1e6,
                           "source_timestamps_ns": np.asarray(stamps, dtype=np.int64)[indices].tolist()}
    def concatenate(kind):
        return np.concatenate([arrays[f"/{side}-{part}-{kind}"] for side in ["left", "right"] for part in ["arm", "ee"]], axis=1).tolist()
    annotations = []
    apath = directory / "annotation.mcap"
    if apath.exists():
        with apath.open("rb") as stream:
            for _, _, message, decoded in make_reader(stream, decoder_factories=[DecoderFactory()]).iter_decoded_messages():
                annotations.append({"timestamp_ns": message.log_time, "time_seconds": (message.log_time - int(clock[0])) / 1e9,
                                    "start_frame": int(np.searchsorted(clock, message.log_time)), "text": str(decoded.data)})
    layout = [key for side in ["left", "right"] for key in [*[f"{side}_joint{i}" for i in range(1, 7)], f"{side}_gripper"]]
    return {
        "dataset": "ABC-130k", "subset": "yam", "profile": "abc_yam", "task": task,
        "states": concatenate("state"), "actions": concatenate("action"), "command_endpoints": concatenate("action"),
        "timestamps": ((clock - clock[0]) / 1e9).tolist(), "timestamps_ns": clock.tolist(),
        "time_basis": "native left-arm action clock; observations aligned causally by absolute MCAP timestamps",
        "state_layout": layout, "action_layout": layout, "end_effector_poses": observed_poses,
        "camera_paths": image_rows, "camera_source_indices": image_indices,
        "source_annotations": annotations, "native_aux": {"channel_counts": counts, "camera_codecs": codecs},
        "source": {"root": str(root), "file": str(path), "sha256": sha256_file(path), "split": choice["episode"].split("/")[1],
                   "annotation_sha256": sha256_file(apath) if apath.exists() else None},
        "alignment": {"source_rows": len(clock), "dropped_action_rows": 0, "streams": alignment},
    }
