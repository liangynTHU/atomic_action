"""Render metadata only after release validation succeeds."""

import json

from .styles import STYLE_SPECS
from .evidence import QUALITY_POLICY


def write_release_metadata(
    release_root,
    report,
    *,
    expected_rows,
    episodes,
    segments,
    cot_quality,
    short_merge_checked,
    corrections_count,
):
    (release_root / "FINAL_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_report = {
        "schema_version": "robotwin_cot_release_v1",
        "quality_policy": QUALITY_POLICY,
        "created_at_utc": report["created_at_utc"],
        "passed": True,
        "counts": report["counts"],
        "quality_contract": report["quality"],
        "artifact_hashes": report["hashes"],
        "outputs": {
            "vlm_train": "data/vlm/train.jsonl",
            "vla_train": "data/vla/train.json",
            "action_segmentation": "data/action_segmentation/train.jsonl",
            "source_manifest": "metadata/source_manifest.jsonl",
            "action_label_corrections": "metadata/ACTION_LABEL_CORRECTIONS.jsonl",
            "cot_style_report": "COT_STYLE_REPORT.json",
            "final_audit": "FINAL_AUDIT.json",
            "annotations": "work/cot5_annotations.jsonl",
            "request_journals": "work/REQUEST_JOURNALS.json",
            **(
                {"cot_quality_audit": "COT_QUALITY_AUDIT.json"}
                if cot_quality is not None
                else {}
            ),
            **(
                {
                    "short_segment_mapping": ("metadata/SHORT_SEGMENT_MAPPING.jsonl"),
                    "short_segment_merge_report": ("SHORT_SEGMENT_MERGE_REPORT.json"),
                }
                if short_merge_checked
                else {}
            ),
        },
    }
    (release_root / "FINAL_REPORT.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "robotwin_cot_manifest_summary_v1",
        "created_at_utc": report["created_at_utc"],
        "episodes": episodes,
        "segments": segments,
        "tasks": report["counts"]["tasks"],
        "variants": report["counts"]["variants"],
        "five_cot_styles": [spec["name"] for spec in STYLE_SPECS],
        "action_segmentation_manifest": "data/action_segmentation/train.jsonl",
        "source_manifest": "metadata/source_manifest.jsonl",
        "recommended_outputs": {
            "vlm": "data/vlm/train.jsonl",
            "vla": "data/vla/train.json",
            "action_segmentation": "data/action_segmentation/train.jsonl",
        },
    }
    (release_root / "metadata/MANIFEST_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    short_merge_note = (
        "- Shorter-than-8-row segments adjacent to an identical action "
        "pair were merged; different-action transitions were preserved.\n\n"
        if short_merge_checked
        else ""
    )
    readme = (
        "# RoboTwin normal-success CoT release\n\n"
        "This is the only supported normal-success RoboTwin CoT release.\n\n"
        f"- `data/vlm/train.jsonl`: {segments:,} thinking-mode VLM rows.\n"
        "- `data/vla/train.json`: aligned action-only VLA episodes.\n"
        "- `data/action_segmentation/train.jsonl`: recovery-shaped action "
        "segmentation, one episode per line with a complete `segments` array.\n"
        "- `metadata/source_manifest.jsonl`: byte-identical provenance copy of "
        "the action segmentation file.\n"
        "- Each row passes a target-blind observer, a visual writer, and "
        "independent grounding and action critics using the actual student images. "
        "Object names, colors, spatial relations and visible contact are allowed "
        "when supported by those images. Planned actions are separated from observed facts.\n"
        "- Original request journals and portable receipts are included under `work/`.\n"
        "- Unresolved evidence/target conflicts block release; no fallback prose is used.\n"
        "- Five near-balanced reasoning styles are used. Every CoT ends "
        "with the programmatically fixed action sentence and final four-field "
        "JSON.\n"
        f"- {corrections_count} raw-trajectory-confirmed action-label corrections are applied.\n"
        "- The single/multi system prompts match the documented planner response-"
        "mode contract. History prompts distinguish valid action rows, aligned "
        "slots, and padding slots.\n\n"
        + short_merge_note
        + "See `FINAL_REPORT.json`, `FINAL_AUDIT.json`, and "
        "`COT_STYLE_REPORT.json`.\n"
    )
    (release_root / "README.md").write_text(readme, encoding="utf-8")
    (release_root / "control").mkdir(parents=True, exist_ok=True)
    (release_root / "control/COMPLETE").write_text(
        report["created_at_utc"] + "\n", encoding="utf-8"
    )
