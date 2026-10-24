#!/usr/bin/env python3
"""Full original episode-length census for the local OXE RLDS collection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_seg.common.artifacts import atomic_write_json
from atomic_seg.common.locking import writer_lock
from atomic_seg.pretrain.episode_lengths import inventory, scan_shard, summarize


def finalize(output, datasets, jobs, results, elapsed):
    grouped = defaultdict(Counter)
    anomalies = defaultdict(Counter)
    for result in results:
        key = result["dataset"]
        counts = {int(k): v for k, v in result["histogram"].items()}
        grouped[key].update(counts)
        grouped[(key, result["split"])].update(counts)
        grouped["ALL"].update(counts)
        anomalies[key].update(result["anomalies"])
    report = {"complete": len(results) == len(jobs), "elapsed_seconds": elapsed,
              "datasets": [], "all": summarize(grouped["ALL"]), "scanned_shards": len(results),
              "expected_shards": len(jobs), "count_basis": "original RLDS steps per complete TFRecord Example",
              "validation": "every record: is_first/is_last/is_terminal lengths agree with native action scalar count; shard record counts agree with metadata",
              "not_performed": "no image decoding; no whole-payload CRC validation for skipped bytes; no claim about success or effective nonpadding commands",
              "segmentation_capacity": "floor(original_steps/minimum_segment_steps) is a length-only upper bound, not a semantic segmentation result"}
    for dataset in datasets:
        name = dataset["dataset"]
        item = {**dataset, **summarize(grouped[name]), "boundary_flag_anomalies": dict(anomalies[name]),
                "by_split": {split["name"]: summarize(grouped[(name, split["name"])]) for split in dataset["splits"]}}
        assert item["episodes"] == sum(s["episodes"] for s in dataset["splits"])
        report["datasets"].append(item)
    atomic_write_json(output / "REPORT.json", report)
    columns = ["dataset", "episodes", "total_steps", "minimum", "p10", "median", "p90", "maximum", "mean",
               "below_16_pct", "below_32_pct", "below_64_pct", "below_96_pct", "mean_max_segments_at_min16", "mean_max_segments_at_min32"]
    with (output / "dataset_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["datasets"])
    lines = ["# OXE 原始 episode 帧数全量统计", "", "统计原始 RLDS step 行数，未做原子分段、抽帧、重复帧或 padding。各子集采样率不同，不据此直接比较秒数。", "",
             "| 子数据集 | episode 数 | 最少 | P10 | 中位数 | P90 | 最多 | <32 帧 | <64 帧 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["datasets"]:
        lines.append(f"| {r['dataset']} | {r['episodes']:,} | {r['minimum']} | {r['p10']:g} | {r['median']:g} | {r['p90']:g} | {r['maximum']} | {r['below_32_pct']:.1f}% | {r['below_64_pct']:.1f}% |")
    lines += ["", "16/32 帧分段容量是 floor(episode_steps/16 或 32) 的数学上限，真实语义划分可能更少。原始终止 step 仍计入；不把 is_terminal 等同于任务成功，也未逐动作判定有效命令。", "", "完整分布、各 split 和边界标志异常见 REPORT.json；逐 shard 统计见 shards.jsonl。扫描跳过图片、深度等大 payload，仅核对所需 protobuf 字段、TFRecord 长度 CRC、边界/动作长度及 metadata episode 数。"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    output = args.output.resolve()
    with writer_lock(output / "control/stats.lock"):
        datasets, jobs = inventory(args.root.resolve())
        if args.dataset:
            datasets = [d for d in datasets if d["dataset"] in args.dataset]
            jobs = [j for j in jobs if j["dataset"] in args.dataset]
        atomic_write_json(output / "inventory.json", {"root": str(args.root.resolve()), "datasets": datasets, "shards": len(jobs),
                                                    "expected_episodes": sum(j["expected_episodes"] for j in jobs)})
        journal = output / "shards.jsonl"
        prior = []
        if journal.exists():
            if not args.resume:
                raise FileExistsError("use --resume for existing stats")
            prior = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
        done = {r["path"]: r for r in prior}
        for path, result in done.items():
            stat = Path(path).stat()
            if stat.st_size != result["file_bytes"] or stat.st_mtime_ns != result["mtime_ns"]:
                raise ValueError("scanned source shard changed")
        pending = [j for j in jobs if j["path"] not in done]
        results = list(prior)
        started = last = time.monotonic()
        with journal.open("a", buffering=1) as stream, ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(scan_shard, job): job for job in pending}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    atomic_write_json(output / "ERROR.json", {"job": job, "error": str(error)})
                    raise
                stream.write(json.dumps(result, separators=(",", ":")) + "\n")
                results.append(result)
                if time.monotonic() - last >= 10 or len(results) == len(jobs):
                    progress = {"shards": len(results), "expected_shards": len(jobs), "episodes": sum(r["actual_episodes"] for r in results),
                                "elapsed_seconds": time.monotonic() - started, "last_dataset": job["dataset"]}
                    atomic_write_json(output / "progress.json", progress)
                    print(json.dumps(progress), flush=True)
                    last = time.monotonic()
        report = finalize(output, datasets, jobs, results, time.monotonic() - started)
        print(json.dumps({"complete": report["complete"], "episodes": report["all"]["episodes"], "datasets": len(datasets),
                          "report": str(output / "REPORT.json")}), flush=True)


if __name__ == "__main__":
    main()
