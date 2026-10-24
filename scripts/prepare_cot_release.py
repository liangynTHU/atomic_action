#!/usr/bin/env python3
"""Materialize a release in its own directory without installing it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import shutil

from atomic_seg.cot_data.generation import finalize_dataset
from atomic_seg.cot_data.provenance import bundle_annotations
from atomic_seg.common.artifacts import atomic_write_json, sha256_file
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input_root.resolve(), args.output_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("input and release roots must be separate directories")
    # A stale success marker must not survive a failed materialization.
    (output / "control/COMPLETE").unlink(missing_ok=True)
    finalize_dataset(
        input_root=source,
        output_root=output,
        annotation_path=args.annotations,
        overwrite=True,
        require_modulo_style_allocation=False,
    )
    for name in (
        "metadata/ACTION_LABEL_CORRECTIONS.jsonl",
        "metadata/SHORT_SEGMENT_MAPPING.jsonl",
        "SHORT_SEGMENT_MERGE_REPORT.json",
        "ACTION_MANIFEST_REPORT.json",
        "SEMANTIC_CLEANUP_REPORT.json",
    ):
        if (source / name).is_file():
            (output / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, output / name)
    corrections = output / "metadata/ACTION_LABEL_CORRECTIONS.jsonl"
    if not corrections.exists():
        corrections.touch()
    (output / "data/action_segmentation").mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        output / "metadata/source_manifest.jsonl",
        output / "data/action_segmentation/train.jsonl",
    )
    (output / "work").mkdir(exist_ok=True)
    bundled = output / "work/cot5_annotations.jsonl"
    bundle_annotations(args.annotations, bundled)
    report_path = output / "COT_STYLE_REPORT.json"
    report = json.loads(report_path.read_text())
    report["hashes"]["annotations"] = sha256_file(bundled)
    report["hashes"]["work/REQUEST_JOURNALS.json"] = sha256_file(
        output / "work/REQUEST_JOURNALS.json"
    )
    atomic_write_json(report_path, report)


if __name__ == "__main__":
    main()
