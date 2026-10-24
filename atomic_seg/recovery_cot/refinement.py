"""Rebuild an old recovery release through the mandatory current action policy.

Old annotations are never seeded: changed boundaries change origins, history,
chunk budgets and style allocation. The original native audit is read-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atomic_seg.common.artifacts import atomic_write_json, sha256_file
from .contracts import DEFAULT_OUTPUT_ROOT as _FORMAL_ROOT
from .native import create_manifest, manifest_paths
from .policy import QUALITY_POLICY

DEFAULT_INPUT_ROOT = _FORMAL_ROOT
DEFAULT_OUTPUT_ROOT = _FORMAL_ROOT.with_name("robotwin_recovery_cot_rebuilt")


def prepare_still_refinement(
    input_root: Path,
    output_root: Path,
    *,
    workers: int = 16,
) -> dict[str, Any]:
    source = input_root.resolve()
    destination = output_root.resolve()
    if destination == source or source in destination.parents:
        raise ValueError(
            "rebuild output must be separate from the old recovery release"
        )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"rebuild output is not empty: {destination}")
    summary_path = manifest_paths(source)[1]
    summary = json.loads(summary_path.read_text())
    source_root = Path(summary["source_root"])
    if not source_root.is_absolute():
        raise ValueError(
            "old manifest source_root must be an absolute native dataset path"
        )
    rebuilt = create_manifest(source_root, destination, workers=workers)
    report = {
        "quality_policy": QUALITY_POLICY,
        "input_root": str(source),
        "source_root": str(source_root),
        "output_root": str(destination),
        "old_manifest_sha256": sha256_file(manifest_paths(source)[0]),
        "counts": {
            "input_blocks": summary["blocks"],
            "output_blocks": rebuilt["blocks"],
            "output_episodes": rebuilt["episodes"],
            "reused_annotations": 0,
        },
        "policy": "rebuild from full current native production audit; generate all CoT anew",
        "next_step": "recovery_cot.py run-rank for each rank, then finalize (docs/PIPELINES.md)",
    }
    atomic_write_json(destination / "SEGMENTATION_REFINEMENT_REPORT.json", report)
    return report
