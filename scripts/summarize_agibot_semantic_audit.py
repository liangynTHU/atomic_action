#!/usr/bin/env python3
"""Join explicit visual-review decisions with every segment and verify exports."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atomic_seg.pretrain.realomni.storage import atomic_bytes, atomic_json, file_hash, read_json, verify_directory


def summarize(evidence_path, decisions_path):
    evidence, decisions = read_json(evidence_path), read_json(decisions_path)
    if evidence.get("partial_accepted_batches"):
        raise ValueError("final audit summary requires committed exports, not partial accepted batches")
    if decisions.get("review_complete") is False:
        raise ValueError("cannot summarize an incomplete manual review")
    if "reviewed_episode_segment_counts" in decisions:
        coverage = Counter(row["episode_id"] for row in evidence["rows"])
        if dict(coverage) != decisions["reviewed_episode_segment_counts"]:
            raise ValueError("explicit review coverage does not match the committed evidence")
    output = evidence_path.resolve().parent
    run = Path(evidence["source_run"])
    if output == run or run in output.parents or output in run.parents:
        raise ValueError("audit output must not overlap source run")
    by_key = {(item["episode_id"], item["display_segment"]): item for item in decisions["findings"]}
    if len(by_key) != len(decisions["findings"]):
        raise ValueError("duplicate decisions")
    rows = []
    episode_counts = defaultdict(Counter)
    categories, totals = Counter(), Counter()
    for row in evidence["rows"]:
        key = row["episode_id"], row["display_segment"]
        if not Path(row["evidence_sheet"]).is_file() or not all(Path(o["path"]).is_file() for o in row["observations"]):
            raise ValueError("missing visual evidence")
        decision = by_key.pop(key, {"status": "no_clear_issue", "categories": [],
                                   "reason": decisions["default_reason"]})
        if decision["status"] not in {"confirmed_issue", "uncertain", "no_clear_issue"}:
            raise ValueError("unknown review decision")
        if decision["status"] != "no_clear_issue" and not decision.get("reason"):
            raise ValueError("missing review rationale")
        merged = {**row, **decision, "reviewed": True,
                  "method": "all five chronological teacher observations plus named measured-arm signals"}
        rows.append(merged)
        status = decision["status"]
        totals[status] += 1
        episode_counts[row["episode_id"]][status] += 1
        if status == "confirmed_issue":
            categories.update(set(decision["categories"]))
    if by_key:
        raise ValueError(f"decisions do not match evidence: {sorted(by_key)}")
    exports = [json.loads(line) for line in (run / "accepted.jsonl").read_text().splitlines()]
    for entry in exports:
        path = Path(entry["path"])
        verify_directory(path, read_json(path / "COMMIT.json")["identity"])
        if file_hash(path / "vla.json") != evidence["source_export_hashes"][entry["episode_id"]]:
            raise ValueError("source labels changed during visual review")
    summary = {"scope": decisions["scope"], "episodes": evidence["episodes"], "segments": len(rows),
               "counts": dict(totals), "confirmed_issue_percent": round(100 * totals["confirmed_issue"] / len(rows), 2),
               "episodes_with_confirmed_issues": sum(bool(c["confirmed_issue"]) for c in episode_counts.values()),
               "category_segment_counts": dict(categories), "categories_may_overlap": True,
               "per_episode": {key: dict(value) for key, value in episode_counts.items()},
               "source_exports_unchanged_and_commits_verified": True,
               "review_limit": "full segment coverage, not continuous inspection of every video frame; no_clear_issue is not proof of zero errors"}
    atomic_json(output / "decisions.json", {"summary": summary, "rows": rows})
    atomic_json(output / "SUMMARY.json", summary)
    lines = ["# AgiBot Semantic Audit", "", decisions["scope"], "",
             f"- Episodes: {evidence['episodes']}; segments reviewed: {len(rows)}.",
             f"- Confirmed issue segments: {totals['confirmed_issue']} ({summary['confirmed_issue_percent']}%).",
             f"- Uncertain segments: {totals['uncertain']}; no clear issue: {totals['no_clear_issue']}.",
             f"- Episodes with confirmed issues: {summary['episodes_with_confirmed_issues']}.",
             "- Each segment counts once; category counts can overlap.",
             "- Export commits and label hashes verified unchanged.",
             "- All five sampled observations reviewed; unsampled transient events can be missed.", "",
             "| Episode | Total | Confirmed | Uncertain |", "| --- | ---: | ---: | ---: |"]
    for episode, counts in episode_counts.items():
        lines.append(f"| {episode} | {sum(counts.values())} | {counts['confirmed_issue']} | {counts['uncertain']} |")
    for status, title in (("confirmed_issue", "Confirmed Findings"), ("uncertain", "Uncertain Findings")):
        lines += ["", "## " + title, ""]
        for row in rows:
            if row["status"] != status:
                continue
            sheet = Path(row["evidence_sheet"]).relative_to(output).as_posix()
            lines += [f"### {row['episode_id']} / Segment {row['display_segment']}", "",
                      f"Frames {row['start_frame']}-{row['target_frame']} (last is target observation). "
                      f"Categories: {', '.join(row['categories'])}.", "",
                      row["reason"], "", f"> {row['sub_task']}", "", f"[Evidence]({sheet})", ""]
    atomic_bytes(output / "REPORT.md", ("\n".join(lines) + "\n").encode())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.evidence, args.decisions)


if __name__ == "__main__":
    main()
