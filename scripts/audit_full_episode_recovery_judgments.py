#!/usr/bin/env python3
"""Summarize decisive model outcomes against dataset recovery phases."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

FAILURE_OUTCOMES = {"failed_no_effect", "failed_different"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in args.input.open("r", encoding="utf-8")
        if line.strip()
    ]
    audits: list[dict[str, Any]] = []
    for row in rows:
        reference = str(row["current_phase"])
        status = str(row["current_status"])
        expected_status = str(row["raw_phase"])
        outcome = str(row["previous_subtask_outcome"])
        if outcome not in {*FAILURE_OUTCOMES, "partial", "success"}:
            raise ValueError(f"invalid decisive outcome: {outcome}")
        if status != expected_status:
            disposition = "invalid_status_phase_mismatch"
        elif outcome in FAILURE_OUTCOMES:
            disposition = "decisive_failed_outcome"
        elif outcome == "partial":
            disposition = "decisive_partial_outcome"
        else:
            disposition = "decisive_success_outcome"
        audits.append(
            {
                "sample_id": row["sample_id"],
                "episode_id": row["episode_id"],
                "row_index": row["row_index"],
                "sample_type": row["sample_type"],
                "dataset_phase": reference,
                "raw_phase": row["raw_phase"],
                "current_status": status,
                "model_outcome": outcome,
                "confidence": row["model_judgment"]["confidence"],
                "disposition": disposition,
                "action_supervision": row["action_supervision"],
            }
        )
    report = {
        "schema": "full_episode_recovery_judgment_audit",
        "row_count": len(rows),
        "disposition_counts": dict(Counter(value["disposition"] for value in audits)),
        "status_counts": dict(Counter(value["current_status"] for value in audits)),
        "model_outcome_counts": dict(
            Counter(value["model_outcome"] for value in audits)
        ),
        "rows": audits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["disposition_counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
