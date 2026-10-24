"""Shared JSONL, checksums, atomic output and bounded CPU concurrency."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_jsonl(
    path: Path,
    values: Iterable[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(compact_json(value) + "\n")
    os.replace(temporary, path)


def append_jsonl(stream: Any, value: Mapping[str, Any]) -> None:
    stream.write(compact_json(value) + "\n")
    stream.flush()


def jsonl_rows(
    path: Path,
    *,
    tolerate_truncated_last: bool = False,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if tolerate_truncated_last and index == len(lines) - 1:
                break
            raise
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{index + 1}: expected object")
        output.append(row)
    return output


def repair_jsonl_tail(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    size = path.stat().st_size
    with path.open("rb+") as stream:
        line_number = 0
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return False
            line_number += 1
            at_eof = stream.tell() == size
            if not line.endswith(b"\n"):
                if not at_eof:
                    raise ValueError(
                        f"{path}:{line_number}: unterminated non-final line"
                    )
                stream.truncate(start)
                return True
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not at_eof:
                    raise ValueError(
                        f"{path}:{line_number}: corrupt non-final row"
                    ) from exc
                stream.truncate(start)
                return True
            if not isinstance(value, dict):
                if not at_eof:
                    raise ValueError(f"{path}:{line_number}: non-object JSONL row")
                stream.truncate(start)
                return True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bounded_parallel(
    items: Iterable[Any],
    *,
    workers: int,
    fn: Callable[[Any], Any],
) -> Iterator[tuple[Any, Any | None, BaseException | None]]:
    iterator = iter(items)
    max_in_flight = max(1, workers * 2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        pending: dict[concurrent.futures.Future[Any], Any] = {}

        def fill() -> None:
            while len(pending) < max_in_flight:
                try:
                    item = next(iterator)
                except StopIteration:
                    return
                pending[pool.submit(fn, item)] = item

        fill()
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                item = pending.pop(future)
                try:
                    yield item, future.result(), None
                except BaseException as exc:  # noqa: BLE001
                    yield item, None, exc
            fill()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream strict object rows; missing files and malformed rows are errors."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value
