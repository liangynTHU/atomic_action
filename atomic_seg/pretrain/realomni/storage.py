"""Durable stage records; shared locks are verified on every production node."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import socket
import time
import uuid
from pathlib import Path


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (encode(value) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text())


def iter_jsonl(path):
    with Path(path).open() as f:
        for line in f:
            if not line.endswith("\n"):
                raise ValueError(f"incomplete committed JSONL: {path}")
            yield json.loads(line)


def fingerprint(path):
    p = Path(path).resolve()
    stat = p.stat()
    return {"path": str(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def check_source(row):
    if fingerprint(row["file"]["path"]) != row["file"]:
        raise RuntimeError(f"source changed since inventory: {row['episode_id']}")


@contextlib.contextmanager
def lock(path, *, blocking=False):
    """Never unlink a flock inode. SIGKILL releases the kernel's ownership."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def episode_dir(run, row):
    key = row["key"]
    return Path(run) / "episodes" / key[:2] / key


def status(run, row, version, state, **details):
    path = episode_dir(run, row) / "status.json"
    old = read_json(path) if path.exists() else {}
    if old and (old["version"] != version or old["source_identity"] != row["source_identity"]):
        raise RuntimeError("episode checkpoint identity changed")
    record = {**old, "episode_id": row["episode_id"], "source_identity": row["source_identity"],
              "version": version, "source_file": row["file"]["path"], "logical_shard": row["shard"],
              "status": state, "updated_at": time.time(), **details}
    atomic_json(path, record)
    return record


def commit_directory(temporary, final, identity):
    """A directory is visible only after its files, hashes and COMMIT are durable."""
    temporary, final = Path(temporary), Path(final)
    files = []
    for path in sorted(temporary.rglob("*")):
        if path.is_file():
            with path.open("rb") as f:
                os.fsync(f.fileno())
            files.append({"path": str(path.relative_to(temporary)), "sha256": file_hash(path)})
    atomic_json(temporary / "COMMIT.json", {"identity": identity, "files": files})
    for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()), reverse=True):
        sync_directory(directory)
    sync_directory(temporary)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if read_json(final / "COMMIT.json")["identity"] != identity:
            raise RuntimeError(f"immutable commit collision: {final}")
        verify_directory(final, identity)
        shutil.rmtree(temporary)
        return
    os.rename(temporary, final)
    sync_directory(final.parent)


def verify_directory(path, identity):
    path = Path(path)
    record = read_json(path / "COMMIT.json")
    if record["identity"] != identity:
        raise RuntimeError(f"stage identity changed: {path}")
    expected = {item["path"] for item in record["files"]}
    actual = {str(p.relative_to(path)) for p in path.rglob("*") if p.is_file() and p != path / "COMMIT.json"}
    if len(expected) != len(record["files"]) or actual != expected:
        raise RuntimeError(f"committed artifact file set mismatch: {path}")
    for item in record["files"]:
        if Path(item["path"]).is_absolute() or ".." in Path(item["path"]).parts:
            raise RuntimeError("invalid committed relative path")
        if file_hash(path / item["path"]) != item["sha256"]:
            raise RuntimeError(f"committed artifact checksum mismatch: {path / item['path']}")
    return record


def staging(parent, name):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / (".pending_" + name + "_" + uuid.uuid4().hex)
    path.mkdir()
    return path


class Journal:
    """Per-episode request journal. Salvage only a torn final record on restart."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open("r+b") as f:
                while line := f.readline():
                    if not line.endswith(b"\n"):
                        atomic_bytes(self.path.with_name(self.path.name + ".torn." + uuid.uuid4().hex), line)
                        f.seek(-len(line), os.SEEK_CUR)
                        f.truncate()
                        f.flush()
                        os.fsync(f.fileno())
                        break
                    json.loads(line)

    def append(self, value):
        existed = self.path.exists()
        with self.path.open("ab") as f:
            f.write((encode(value) + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        if not existed:
            sync_directory(self.path.parent)


def process_identity():
    return {"hostname": socket.gethostname(), "pid": os.getpid(),
            "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]}
