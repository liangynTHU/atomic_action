"""Bundle request journals with portable annotation receipts."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

from atomic_seg.common.artifacts import (
    atomic_write_json, atomic_write_jsonl, iter_jsonl, sha256_file,
)


def receipts(value):
    if isinstance(value, dict):
        if {"journal", "byte_offset", "bytes", "sha256"} <= set(value):
            yield value
        else:
            for child in value.values():
                yield from receipts(child)
    elif isinstance(value, list):
        for child in value:
            yield from receipts(child)


def bundle_annotations(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    files, replacements = {}, {}
    for annotation in iter_jsonl(source):
        for receipt in receipts(annotation):
            original = receipt["journal"]
            if original in replacements:
                continue
            journal = Path(original)
            if not journal.is_absolute():
                journal = source.parent / journal
            checksum = sha256_file(journal)
            name = f"requests/{checksum}.jsonl"
            target = destination.parent / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.resolve() != journal.resolve():
                shutil.copy2(journal, target)
            if sha256_file(target) != checksum:
                raise ValueError("Request journal changed during bundling")
            files[name] = checksum
            replacements[original] = name

    def converted():
        for annotation in iter_jsonl(source):
            result = copy.deepcopy(annotation)
            for receipt in receipts(result):
                receipt["journal"] = replacements[receipt["journal"]]
            yield result

    atomic_write_jsonl(destination, converted())
    manifest = {"schema_version": "robotwin_request_journals_v1", "files": files}
    atomic_write_json(destination.parent / "REQUEST_JOURNALS.json", manifest)
    return manifest
