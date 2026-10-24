#!/usr/bin/env python3
"""Build an HTML review page for generated recovery CoT rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


from atomic_seg.common.rendering import escape as _escape
from atomic_seg.common.rendering import image_card


def main() -> None:
    args = parse_args()
    rows = []
    with args.input.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
            if args.limit is not None and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError("recovery CoT preview input contains no rows")
    input_root = args.input.parent.resolve()
    output_root = args.output.parent.resolve()
    cards = []
    for row in rows:
        figures = []
        for role, relative in zip(row["image_roles"], row["images"]):
            source = (input_root / relative).resolve()
            href = Path(os.path.relpath(source, output_root)).as_posix()
            figures.append(image_card(href, role))
        previous = json.dumps(
            row["previous_outcome"],
            ensure_ascii=False,
            indent=2,
        )
        target = json.dumps(row["target"], ensure_ascii=False, indent=2)
        human = row["conversations"][0]["value"]
        assistant = row["conversations"][1]["value"]
        cards.append(
            f"""
            <article>
              <header>
                <div>
                  <h2>{_escape(row["episode_id"])} / segment
                    {_escape(row["segment_index"])}</h2>
                  <p>{_escape(row["task"])}</p>
                </div>
                <span class="badge {_escape(row["previous_outcome"]["status"])}">
                  previous: {_escape(row["previous_outcome"]["status"])}
                </span>
              </header>
              <p class="meta">L{_escape(row["recovery_level"])}
                {_escape(row.get("recovery_level_name", ""))} ·
                {_escape(row["failure_type"])}</p>
              <div class="images">{"".join(figures)}</div>
              <div class="columns">
                <section><h3>Previous-outcome audit</h3>
                  <pre>{_escape(previous)}</pre></section>
                <section><h3>Current target</h3>
                  <pre>{_escape(target)}</pre></section>
              </div>
              <details><summary>Training human input</summary>
                <pre>{_escape(human)}</pre></details>
              <section><h3>Assistant CoT + action</h3>
                <pre class="assistant">{_escape(assistant)}</pre></section>
            </article>
            """
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Recovery CoT pilot review</title>
  <style>
    :root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
    body {{ margin: 0; padding: 24px; background: #10131a; color: #edf2f7; }}
    main {{ max-width: 1500px; margin: auto; }}
    article {{ background: #1a202c; border: 1px solid #344054;
      border-radius: 14px; padding: 20px; margin: 0 0 24px; }}
    header {{ display: flex; justify-content: space-between; gap: 16px;
      align-items: flex-start; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .meta {{ color: #aab6c5; }}
    .badge {{ border-radius: 999px; padding: 7px 11px; font-weight: 700;
      white-space: nowrap; background: #334155; }}
    .badge.success {{ background: #166534; }}
    .badge.failed_no_effect, .badge.failed_different {{ background: #9f1239; }}
    .badge.partial, .badge.unknown {{ background: #92400e; }}
    .images {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px; margin: 18px 0; }}
    figure {{ margin: 0; background: #0f172a; padding: 8px; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; border-radius: 5px; }}
    figcaption {{ margin-top: 6px; color: #cbd5e1; font-size: 12px; }}
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    section, details {{ background: #111827; border-radius: 8px; padding: 12px;
      margin-top: 12px; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; margin: 0;
      font-size: 12px; line-height: 1.45; }}
    .assistant {{ font-size: 14px; }}
    summary {{ cursor: pointer; font-weight: 700; margin-bottom: 10px; }}
    @media (max-width: 900px) {{
      .images, .columns {{ grid-template-columns: 1fr; }}
      header {{ flex-direction: column; }}
    }}
  </style>
</head>
<body><main>
  <h1>Recovery CoT pilot review</h1>
  <p>{len(rows)} segment rows. Audit labels are shown on this review page but
  are hidden from each training human prompt.</p>
  {"".join(cards)}
</main></body></html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"rows={len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
