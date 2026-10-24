#!/usr/bin/env python3
"""Build an episode-grouped HTML preview for full recovery CoT data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os
import shutil
import zipfile
from collections import defaultdict
from typing import Any, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--judgment-manifest", type=Path, default=None)
    parser.add_argument("--episode-review-dir", type=Path, default=None)
    parser.add_argument("--audit-report", type=Path, default=None)
    return parser.parse_args()


from atomic_seg.common.rendering import escape as _escape
from atomic_seg.common.rendering import image_card


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError("preview input contains no rows")
    return rows


def _relative_asset(
    input_root: Path,
    output_page: Path,
    value: str,
) -> str:
    source = (input_root / value).resolve()
    return Path(os.path.relpath(source, output_page.parent)).as_posix()


def _row_card(
    row: Mapping[str, Any],
    *,
    input_root: Path,
    output_page: Path,
) -> str:
    figures = []
    for role, frame, relative in zip(
        row["image_roles"],
        row["image_frames"],
        row["images"],
    ):
        figures.append(
            image_card(
                _relative_asset(input_root, output_page, relative),
                f"{role} · frame {frame}",
            )
        )
    judgment = json.dumps(
        row["model_judgment"],
        ensure_ascii=False,
        indent=2,
    )
    reference_failure = json.dumps(
        row.get("reference_failure_context"),
        ensure_ascii=False,
        indent=2,
    )
    action = json.dumps(
        row["next_action_target"],
        ensure_ascii=False,
        indent=2,
    )
    candidate_action = json.dumps(
        row.get("candidate_next_action"),
        ensure_ascii=False,
        indent=2,
    )
    assistant = row["messages"][2]["content"]
    human = row["messages"][1]["content"]
    action_badge = "action target" if row["action_supervision"] else "diagnosis only"
    phase = str(row["current_phase"])
    raw_phase = str(row["raw_phase"])
    status = str(row["current_status"])
    phase_sequence = " → ".join(row.get("raw_phase_sequence", [raw_phase]))
    return f"""
    <article class="row {_escape(phase)}">
      <header>
        <div><h2>Row {row["row_index"]} · {_escape(row["previous_segment_id"])}</h2>
          <p>{_escape(row["previous_subtask"])}</p></div>
        <div class="badges">
          <span>{_escape(row["sample_type"])}</span>
          <span class="status">current status: {_escape(status)}</span>
          <span>coarse phase: {_escape(phase)}</span>
          <span>raw phase: {_escape(raw_phase)}</span>
          <span>phase span: {_escape(phase_sequence)}</span>
          <span class="outcome">Qwen outcome: {_escape(row["previous_subtask_outcome"])}</span>
          <span>{_escape(action_badge)}</span>
        </div>
      </header>
      <div class="images">{"".join(figures)}</div>
      <div class="columns">
        <section><h3>Qwen outcome judgment (status comes from dataset)</h3>
          <pre>{_escape(judgment)}</pre>
          <h3>Dataset failure context</h3>
          <pre>{_escape(reference_failure)}</pre></section>
        <section><h3>Eligible next action target</h3>
          <pre>{_escape(action)}</pre>
          <h3>Reference candidate action</h3>
          <pre>{_escape(candidate_action)}</pre>
          <p>Gate: {_escape(row.get("action_supervision_reason", ""))}</p>
        </section>
      </div>
      <details><summary>Model-visible user input</summary>
        <pre>{_escape(human)}</pre></details>
      <section><h3>Phase-aware CoT and structured output</h3>
        <pre class="assistant">{_escape(assistant)}</pre></section>
    </article>
    """


def _episode_page(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_root: Path,
    output_page: Path,
) -> None:
    episode_id = str(rows[0]["episode_id"])
    task = str(rows[0]["task"])
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["current_phase"])] += 1
    cards = "".join(
        _row_card(row, input_root=input_root, output_page=output_page) for row in rows
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape(episode_id)} CoT review</title>
<style>
:root{{color-scheme:dark;font-family:system-ui,sans-serif}}
body{{margin:0;padding:22px;background:#0b1020;color:#eef2ff}}
main{{max-width:1550px;margin:auto}}a{{color:#93c5fd}}
article{{border:1px solid #334155;border-left:6px solid #2563eb;
background:#151c2f;border-radius:12px;padding:18px;margin:18px 0}}
article.failure{{border-left-color:#dc2626}}article.recovery{{border-left-color:#16a34a}}
header{{display:flex;justify-content:space-between;gap:15px;align-items:flex-start}}
h1,h2,h3{{margin-top:0}}.badges{{display:flex;gap:6px;flex-wrap:wrap;
justify-content:flex-end}}.badges span{{background:#334155;border-radius:999px;
padding:5px 9px;font-size:12px}}.images{{display:grid;
grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin:14px 0}}
figure{{margin:0;background:#0f172a;padding:6px;border-radius:7px}}
img{{width:100%;display:block;border-radius:4px}}figcaption{{font-size:11px;
color:#cbd5e1;margin-top:5px}}.columns{{display:grid;grid-template-columns:1fr 1fr;
gap:10px}}section,details{{background:#111827;border-radius:8px;padding:12px;
margin-top:10px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;
font-size:12px;line-height:1.45}}.assistant{{font-size:13px}}
summary{{cursor:pointer;font-weight:700;margin-bottom:8px}}
@media(max-width:900px){{.images,.columns{{grid-template-columns:1fr}}
header{{flex-direction:column}}}}
</style></head><body><main><p><a href="../index.html">← all episodes</a></p>
<h1>{_escape(episode_id)}</h1><p>{_escape(task)}</p>
<p>{len(rows)} rows · rollout {counts["rollout"]} · failure {counts["failure"]} ·
recovery {counts["recovery"]}</p>
<p><code>current_phase</code> is the authoritative label mapped from the
episode Parquet <code>phase</code> column. <code>current_status</code> preserves
the meaningful outcome states <code>failure_outcome</code> and
<code>recovery_outcome</code>, while <code>current_phase</code> provides the
coarse three-way grouping. Qwen judges the previous subtask outcome with an
explicit non-unknown label.</p>{cards}</main></body></html>"""
    output_page.parent.mkdir(parents=True, exist_ok=True)
    output_page.write_text(page, encoding="utf-8")


def _index(
    entries: Sequence[Mapping[str, Any]],
    output: Path,
    *,
    has_complete_episode_review: bool,
) -> None:
    cards = "".join(
        f"""<article><h2>{_escape(entry["episode_id"])}</h2>
        <p>{_escape(entry["task"])}</p><p>{entry["row_count"]} rows ·
        {entry["action_rows"]} action-supervised · {entry["diagnostic_rows"]}
        diagnostic-only</p><a href="episodes/{_escape(entry["slug"])}.html">
        Open CoT review →</a></article>"""
        for entry in entries
    )
    review_link = (
        '<p><a href="complete_episode_review/index.html">Open synchronized '
        "complete-video episode timelines →</a></p>"
        if has_complete_episode_review
        else ""
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Five
full recovery CoT episodes</title><style>:root{{color-scheme:dark;
font-family:system-ui,sans-serif}}body{{margin:0;padding:24px;background:#0b1020;
color:#eef2ff}}main{{max-width:1050px;margin:auto}}article{{background:#151c2f;
border:1px solid #334155;border-radius:12px;padding:17px;margin:14px 0}}
a{{color:#93c5fd}}</style></head><body><main>
<h1>Five full-episode recovery CoT pilots</h1>
<p><code>current_status</code> comes directly from the dataset phase column and
preserves labels such as <code>attempt</code>, <code>failure_outcome</code>, and
<code>recovery_outcome</code>. Qwen supplies only the decisive
<code>previous_subtask_outcome</code>. Source rollout/failure actions are not
imitated, while reference recovery rows carry next-action supervision.</p>
{review_link}{cards}
</main></body></html>"""
    output.write_text(page, encoding="utf-8")


def _package(root: Path, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for value in sorted(root.rglob("*")):
            if value.is_file():
                archive.write(value, Path(root.name) / value.relative_to(root))
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"corrupt ZIP member {corrupt}")
        count = len([name for name in archive.namelist() if not name.endswith("/")])
    return {"path": str(path.resolve()), "files": count, "bytes": path.stat().st_size}


def main() -> None:
    args = parse_args()
    rows = _read_rows(args.input)
    judge_models = sorted(
        {str(row.get("generation", {}).get("judge_model", "unknown")) for row in rows}
    )
    judge_modes = sorted(
        {str(row.get("generation", {}).get("judge_mode", "unknown")) for row in rows}
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["episode_id"])].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_assets = args.input.parent / "assets"
    if not source_assets.is_dir():
        raise FileNotFoundError(source_assets)
    shutil.copytree(
        source_assets,
        args.output_dir / "assets",
        dirs_exist_ok=True,
    )
    shutil.copy2(args.input, args.output_dir / args.input.name)
    for name in (
        "recovery_action_cot.jsonl",
        "recovery_diagnosis_cot.jsonl",
        "judgment_audit.json",
    ):
        source = args.input.parent / name
        if source.is_file():
            shutil.copy2(source, args.output_dir / name)
    source_manifest = args.input.parent / "manifest.json"
    if source_manifest.is_file():
        shutil.copy2(
            source_manifest,
            args.output_dir / "generation_manifest.json",
        )
    if args.judgment_manifest is not None:
        shutil.copy2(
            args.judgment_manifest,
            args.output_dir / "model_judgment_manifest.json",
        )
    if args.audit_report is not None:
        shutil.copy2(
            args.audit_report,
            args.output_dir / "judgment_audit.json",
        )
    has_complete_episode_review = args.episode_review_dir is not None
    if args.episode_review_dir is not None:
        if not args.episode_review_dir.is_dir():
            raise FileNotFoundError(args.episode_review_dir)
        shutil.copytree(
            args.episode_review_dir,
            args.output_dir / "complete_episode_review",
            dirs_exist_ok=True,
        )
    (args.output_dir / "README.md").write_text(
        f"""# Five full-episode recovery CoT pilots

Open `index.html` first.

Contents:

- `full_episode_recovery_cot.jsonl`: 54 training/audit rows from five episodes.
- `recovery_action_cot.jsonl`: rows at dataset-labeled failure/recovery origins
  with a reference corrective action target.
- `recovery_diagnosis_cot.jsonl`: phase/outcome/failure-analysis rows with no
  action target.
- `episodes/*.html`: row-by-row before/after images, dataset phase labels,
  Qwen outcome judgments, phase-aware CoT, and optional recovery targets.
- `complete_episode_review/`: synchronized full videos and complete timelines.
- `model_judgment_manifest.json`: model-authored judgments and provenance.
- `generation_manifest.json`: row/status/outcome counts.

Label policy:

- `current_phase` is one of `rollout`, `failure`, or `recovery` and is mapped
  from the Parquet `phase` column; `raw_phase` preserves values such as
  `attempt`, `failure_observation`, and `recovery_outcome`.
- `previous_subtask_outcome` is inferred from before/after visual and state
  evidence; it is not exposed in the user prompt.
- `current_status` is the raw dataset phase (`rollout`, `attempt`,
  `failure_observation`, `failure_outcome`, `recovery`, etc.); Qwen does not
  predict this field.
- `current_phase` is the coarse three-way mapping used for CoT length and
  phase-level grouping.
- `previous_subtask_outcome` is always decisive; `unknown` is forbidden.
- Source rollout/failure rows are diagnosis-only and do not imitate source
  actions.
- Recovery rows may contain `next_action_target` and action supervision.
- Failure and recovery CoT are intentionally longer than normal-success CoT.

Judge models: {judge_models}

Judge modes: {judge_modes}

The model judgments are preserved with raw-response provenance. Candidate
recovery actions come from the reference recovery trajectory and are supervised
only at dataset-labeled failure/recovery decision origins. Numbered evidence
templates are not inserted into CoT; visual and state observations are fused
into natural failure/recovery analysis.
""",
        encoding="utf-8",
    )
    entries = []
    for index, (episode_id, episode_rows) in enumerate(grouped.items()):
        episode_rows.sort(key=lambda value: int(value["row_index"]))
        slug = f"{index + 1:02d}_{episode_id}"
        page = args.output_dir / "episodes" / f"{slug}.html"
        _episode_page(
            episode_rows,
            input_root=args.output_dir.resolve(),
            output_page=page,
        )
        entries.append(
            {
                "episode_id": episode_id,
                "slug": slug,
                "task": episode_rows[0]["task"],
                "row_count": len(episode_rows),
                "action_rows": sum(
                    bool(row["action_supervision"]) for row in episode_rows
                ),
                "diagnostic_rows": sum(
                    not bool(row["action_supervision"]) for row in episode_rows
                ),
            }
        )
    _index(
        entries,
        args.output_dir / "index.html",
        has_complete_episode_review=has_complete_episode_review,
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "full_episode_recovery_cot_preview",
                "episode_count": len(entries),
                "row_count": len(rows),
                "entries": entries,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    zip_path = args.zip_path or args.output_dir.with_suffix(".zip")
    package = _package(args.output_dir, zip_path)
    print(
        f"episodes={len(entries)} rows={len(rows)} "
        f"zip={package['path']} files={package['files']} bytes={package['bytes']}"
    )


if __name__ == "__main__":
    main()
