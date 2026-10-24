"""Count original RLDS steps in TFRecord episodes without reading image payloads.

The scanner walks protobuf length-delimited fields, seeking over image/depth and
unused fields. Lengths are cross-checked against two boundary flag streams and a
native action feature. It validates framing and length-header CRC32C, but does
not claim to validate the skipped whole-record payload CRC or decode images.
"""

from __future__ import annotations

import json
import math
import struct
import time
from collections import Counter
from pathlib import Path

import crc32c


def read_varint(stream, limit):
    value = 0
    for shift in range(0, 70, 7):
        if stream.tell() >= limit:
            raise ValueError("truncated protobuf varint")
        raw = stream.read(1)
        if not raw:
            raise ValueError("unexpected EOF in protobuf")
        value |= (raw[0] & 127) << shift
        if raw[0] < 128:
            return value
    raise ValueError("invalid protobuf varint")


def fields(stream, end):
    """Yield wire fields while maintaining exact bounds and seeking over payloads."""
    while stream.tell() < end:
        tag = read_varint(stream, end)
        number, wire = tag >> 3, tag & 7
        if not number:
            raise ValueError("invalid protobuf field zero")
        if wire == 2:
            size = read_varint(stream, end)
            start, stop = stream.tell(), stream.tell() + size
        elif wire in (1, 5):
            start = stream.tell()
            stop = start + (8 if wire == 1 else 4)
        elif wire == 0:
            start = stream.tell()
            read_varint(stream, end)
            stop = stream.tell()
            stream.seek(start)
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if stop > end:
            raise ValueError("protobuf field exceeds parent bounds")
        yield number, wire, start, stop
        stream.seek(stop)
    if stream.tell() != end:
        raise ValueError("protobuf message end mismatch")


def int_values(stream, end):
    values = []
    for number, wire, start, stop in fields(stream, end):
        if number != 1:
            continue
        if wire == 2:
            raw = stream.read(stop - start)
            if len(raw) != stop - start:
                raise ValueError("truncated packed int64 list")
            # Flags are normally packed 0/1 bytes; preserve the generic case.
            if all(v < 128 for v in raw):
                values.extend(raw)
            else:
                import io

                packed = io.BytesIO(raw)
                while packed.tell() < len(raw):
                    values.append(read_varint(packed, len(raw)))
        elif wire == 0:
            values.append(read_varint(stream, stop))
        else:
            raise ValueError("unexpected integer-list encoding")
    return values


def feature(stream, end, *, flags=False):
    result = None
    for kind, wire, _, stop in fields(stream, end):
        if kind not in (1, 2, 3) or wire != 2:
            raise ValueError("unexpected Feature encoding")
        if result is not None:
            raise ValueError("multiple Feature kinds")
        if kind == 3:
            values = int_values(stream, stop)
            result = values if flags else len(values)
        elif kind == 2:
            if flags:
                raise ValueError("boundary flags are not int64")
            count = 0
            for number, list_wire, start, finish in fields(stream, stop):
                if number != 1:
                    continue
                if list_wire == 2 and (finish - start) % 4 == 0:
                    count += (finish - start) // 4
                elif list_wire == 5:
                    count += 1
                else:
                    raise ValueError("unexpected float-list encoding")
            result = count
        else:
            raise ValueError("selected scalar/action feature is bytes-encoded")
    if result is None:
        raise ValueError("empty Feature")
    return result


def episode_features(stream, end, selected):
    found = {}
    for number, wire, _, features_end in fields(stream, end):
        if number != 1 or wire != 2:
            raise ValueError("expected tf.train.Example.features")
        for entry_number, entry_wire, _, entry_end in fields(stream, features_end):
            if entry_number != 1 or entry_wire != 2:
                raise ValueError("unexpected features map entry")
            key, payload = None, None
            for field_number, field_wire, start, stop in fields(stream, entry_end):
                if field_wire != 2:
                    raise ValueError("unexpected map key/value encoding")
                if field_number == 1:
                    if stop - start > 4096:
                        raise ValueError("feature key too long")
                    key = stream.read(stop - start).decode("utf-8")
                elif field_number == 2:
                    payload = (start, stop)
            if key in selected:
                if key in found or payload is None:
                    raise ValueError("duplicate/missing selected feature")
                stream.seek(payload[0])
                found[key] = feature(stream, payload[1], flags=key in {"steps/is_first", "steps/is_last", "steps/is_terminal"})
    if set(found) != set(selected):
        raise ValueError(f"missing selected keys: {set(selected) - set(found)}")
    return found


def action_feature(features_path):
    info = json.loads(Path(features_path).read_text())
    steps = info["featuresDict"]["features"]["steps"]["sequence"]["feature"]["featuresDict"]["features"]

    def leaves(value, key):
        if "featuresDict" in value:
            for name, child in value["featuresDict"]["features"].items():
                yield from leaves(child, key + "/" + name)
        elif "tensor" in value:
            tensor = value["tensor"]
            dimensions = [int(v) for v in tensor.get("shape", {}).get("dimensions", [])]
            if all(v > 0 for v in dimensions) and tensor["dtype"] in {"float32", "float64", "int32", "int64", "bool"}:
                yield key, math.prod(dimensions), tensor["dtype"]

    candidates = list(leaves(steps["action"], "steps/action"))
    if not candidates:
        raise ValueError("no supported native action feature")
    # Prefer floating values so their count requires only protobuf lengths.
    return min(candidates, key=lambda value: (not value[2].startswith("float"), value[1], value[0]))


def scan_shard(job):
    path = Path(job["path"])
    length_histogram, anomalies = Counter(), Counter()
    examples, anomaly_examples = [], []
    selected = {"steps/is_first", "steps/is_last", "steps/is_terminal", job["action_key"]}
    started = time.perf_counter()
    size = path.stat().st_size
    count = 0
    with path.open("rb", buffering=8192) as stream:
        while stream.tell() < size:
            offset = stream.tell()
            header = stream.read(12)
            if len(header) != 12:
                raise ValueError(f"{path}: truncated TFRecord header")
            length, checksum = struct.unpack("<QI", header)
            crc = crc32c.crc32c(header[:8])
            masked = (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF
            if masked != checksum:
                raise ValueError(f"{path}:{offset}: length CRC mismatch")
            end = stream.tell() + length
            if end + 4 > size:
                raise ValueError(f"{path}:{offset}: truncated TFRecord payload")
            values = episode_features(stream, end, selected)
            first, last, terminal = (values[key] for key in ("steps/is_first", "steps/is_last", "steps/is_terminal"))
            n = len(last)
            if n == 0 or len(first) != n or len(terminal) != n:
                raise ValueError(f"{path}:{offset}: inconsistent step flag counts")
            if values[job["action_key"]] != n * job["action_width"]:
                raise ValueError(f"{path}:{offset}: action/step length mismatch")
            flags = []
            if first != [1] + [0] * (n - 1):
                flags.append("noncanonical_is_first")
            if last != [0] * (n - 1) + [1]:
                flags.append("noncanonical_is_last")
            if any(x not in (0, 1) for x in terminal):
                flags.append("nonboolean_is_terminal")
            if any(terminal[:-1]):
                flags.append("terminal_before_final_step")
            anomalies.update(flags)
            if flags and len(anomaly_examples) < 3:
                anomaly_examples.append({"record_index": count, "offset": offset, "steps": n, "issues": flags})
            length_histogram[n] += 1
            if count == 0 or n < 16 and len(examples) < 3:
                examples.append({"record_index": count, "offset": offset, "steps": n})
            stream.seek(end + 4)
            count += 1
    if count != job["expected_episodes"]:
        raise ValueError(f"{path}: {count} records, expected {job['expected_episodes']}")
    return {**job, "actual_episodes": count, "file_bytes": size, "mtime_ns": path.stat().st_mtime_ns,
            "seconds": time.perf_counter() - started,
            "histogram": dict(length_histogram), "anomalies": dict(anomalies),
            "examples": examples, "anomaly_examples": anomaly_examples}


def inventory(root):
    jobs, datasets = [], []
    for dataset in sorted(Path(root).iterdir()):
        if not dataset.is_dir():
            continue
        for info_path in sorted(dataset.glob("*/dataset_info.json")):
            info = json.loads(info_path.read_text())
            action_key, width, dtype = action_feature(info_path.with_name("features.json"))
            splits = []
            for split in info["splits"]:
                lengths = list(map(int, split["shardLengths"]))
                paths = sorted(info_path.parent.glob(f"{info['name']}-{split['name']}.tfrecord-*"))
                if len(paths) != len(lengths):
                    raise ValueError(f"{dataset.name}/{split['name']}: missing or extra shards")
                for index, (path, count) in enumerate(zip(paths, lengths)):
                    expected_suffix = f"-{index:05d}-of-{len(lengths):05d}"
                    if not path.name.endswith(expected_suffix):
                        raise ValueError(f"unexpected shard ordinal: {path}")
                    jobs.append({"dataset": dataset.name, "version": info["version"], "split": split["name"],
                                 "shard_index": index, "path": str(path), "expected_episodes": count,
                                 "action_key": action_key, "action_width": width, "action_dtype": dtype})
                splits.append({"name": split["name"], "episodes": sum(lengths), "shards": len(lengths),
                               "serialized_bytes": int(split["numBytes"])})
            datasets.append({"dataset": dataset.name, "version": info["version"], "splits": splits,
                             "action_key_checked": action_key, "action_width": width})
    return datasets, jobs


def summarize(histogram):
    histogram = Counter({int(k): int(v) for k, v in histogram.items()})
    total = sum(histogram.values())
    if total == 0:
        raise ValueError("empty histogram")
    ordered = sorted(histogram.items())

    def rank(index):
        cumulative = 0
        for value, count in ordered:
            cumulative += count
            if cumulative > index:
                return value
        raise IndexError(index)

    def quantile(q):
        position = (total - 1) * q
        low, high = math.floor(position), math.ceil(position)
        return rank(low) * (high - position) + rank(high) * (position - low) if high != low else float(rank(low))

    bins = [("1-15", 1, 15), ("16-31", 16, 31), ("32-63", 32, 63), ("64-127", 64, 127),
            ("128-255", 128, 255), ("256-511", 256, 511), ("512-1023", 512, 1023), ("1024+", 1024, math.inf)]
    result = {"episodes": total, "total_steps": sum(k * v for k, v in histogram.items()),
              "minimum": ordered[0][0], "p10": quantile(.1), "p25": quantile(.25), "median": quantile(.5),
              "p75": quantile(.75), "p90": quantile(.9), "p95": quantile(.95), "p99": quantile(.99), "maximum": ordered[-1][0],
              "bins": {name: sum(count for value, count in ordered if low <= value <= high) for name, low, high in bins}}
    result["mean"] = result["total_steps"] / total
    for limit in (16, 32, 64, 96, 128):
        count = sum(v for k, v in ordered if k < limit)
        result[f"below_{limit}"] = count
        result[f"below_{limit}_pct"] = count * 100 / total
    result["mean_max_segments_at_min16"] = sum((k // 16) * v for k, v in ordered) / total
    result["mean_max_segments_at_min32"] = sum((k // 32) * v for k, v in ordered) / total
    result["histogram"] = dict(ordered)
    return result
