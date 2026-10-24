#!/usr/bin/env python3
"""Configurable CPU segmentation, normal CoT and native recovery CoT workflows."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import os
import signal
import subprocess

from atomic_seg.common.artifacts import atomic_write_json, sha256_file, utc_now
from atomic_seg.common.locking import writer_lock

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("atomic_actions", "cot_data", "recovery_cot_data", "pretrain_noncot")


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    if config.get("pipeline") not in KINDS:
        raise ValueError(f"pipeline must be one of {KINDS}")
    if config["pipeline"] == "pretrain_noncot":
        from scripts.generate_pretrain_noncot import load_config as load_pretrain

        config = load_pretrain(path)
        config["config_path"] = str(path)
        return config
    for key in (
        "input_root",
        "data_root",
        "manifest",
        "output_root",
        "endpoint_file",
        "annotations",
        "reference_driver",
        "source_root",
    ):
        if config.get(key):
            value = Path(config[key]).expanduser()
            config[key] = str(
                value.resolve()
                if value.is_absolute()
                else (path.parent / value).resolve()
            )
    if not config.get("output_root"):
        raise ValueError("output_root is required")
    kind = config["pipeline"]
    required = {
        "atomic_actions": ("data_root", "manifest"),
        "cot_data": ("input_root", "expected_rows", "expected_episodes"),
        "recovery_cot_data": ("source_root", "reference_driver"),
    }[kind]
    for key in required:
        if key not in config:
            raise ValueError(f"{kind} requires {key}")
    if (
        kind == "recovery_cot_data"
        or (kind == "cot_data" and not config.get("annotations"))
    ) and not config.get("endpoint_file"):
        raise ValueError("endpoint_file is required for model generation")
    for key in (
        "workers",
        "batch_size",
        "expected_rows",
        "expected_episodes",
        "world_size",
        "prepare_workers",
        "generation_workers",
        "max_rounds",
        "job_attempts",
        "max_rank_attempts",
        "max_tokens",
    ):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if "timeout" in config and (
        type(config["timeout"]) not in (int, float) or config["timeout"] <= 0
    ):
        raise ValueError("timeout must be positive")
    if "expected_corrections" in config and (
        type(config["expected_corrections"]) is not int
        or config["expected_corrections"] < 0
    ):
        raise ValueError("expected_corrections must be a nonnegative integer")
    if config.get("endpoint_file"):
        from atomic_seg.common.model_client import normalize_endpoint

        endpoints = [
            normalize_endpoint(x).base_url
            for x in Path(config["endpoint_file"]).read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        ]
        if not endpoints or len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoint_file must contain unique nonempty endpoints")
        if config.get("expected_endpoints", len(endpoints)) != len(endpoints):
            raise ValueError("expected_endpoints differs from endpoint_file")
    if (
        "require_no_short_same_action" in config
        and type(config["require_no_short_same_action"]) is not bool
    ):
        raise ValueError("require_no_short_same_action must be boolean")
    output = Path(config["output_root"])
    if config.get("annotations") and (
        output == Path(config["annotations"])
        or output in Path(config["annotations"]).parents
    ):
        raise ValueError("Input annotations must be outside output_root")
    for key in ("input_root", "data_root", "source_root"):
        if config.get(key):
            source = Path(config[key])
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("output_root must be separate from source directories")
    return config


def build_steps(c):
    out = Path(c["output_root"])

    def command(script, *args):
        return [sys.executable, str(ROOT / "scripts" / script), *map(str, args)]

    def step(name, cmd, outputs):
        return {"name": name, "command": cmd, "outputs": list(map(str, outputs))}

    if c["pipeline"] == "pretrain_noncot":
        return [step(
            "pretrain_noncot",
            command("generate_pretrain_noncot.py", "--config", c["config_path"]),
            [out / "VALIDATION.json", out / "index.html", out / "data/vlm/preview.jsonl", out / "data/vla/preview.json"],
        )]
    if c["pipeline"] == "atomic_actions":
        return [
            step(
                "segment",
                command(
                    "segment_episodes.py",
                    "--data-root",
                    c["data_root"],
                    "--manifest",
                    c["manifest"],
                    "--output-dir",
                    out,
                ),
                [out / "manifest.json"],
            )
        ]
    if c["pipeline"] == "recovery_cot_data":
        world = c.get("world_size", 1)
        steps = [
            step(
                "manifest",
                command(
                    "recovery_cot.py",
                    "manifest",
                    "--source-root",
                    c["source_root"],
                    "--output-root",
                    out,
                ),
                [out / "metadata/source_manifest.jsonl"],
            )
        ]
        steps.append(
            step(
                "ranks",
                command(
                    "run_recovery_workers.py",
                    "--output-root",
                    out,
                    "--world-size",
                    world,
                    "--endpoint-file",
                    c["endpoint_file"],
                    "--reference-driver",
                    c["reference_driver"],
                    "--model",
                    c.get("model", "Qwen/Qwen3.8-Flash-Next"),
                    "--prepare-workers",
                    c.get("prepare_workers", 4),
                    "--generation-workers",
                    c.get("generation_workers", 16),
                    "--max-rank-attempts",
                    c.get("max_rank_attempts", 3),
                ),
                [
                    out / f"shards/reports/rank_{r:02d}.generation.json"
                    for r in range(world)
                ],
            )
        )
        steps.append(
            step(
                "finalize",
                command(
                    "recovery_cot.py",
                    "finalize",
                    "--output-root",
                    out,
                    "--world-size",
                    world,
                    "--reference-driver",
                    c["reference_driver"],
                ),
                [
                    out / "FINAL_REPORT.json",
                    out / "COT_QUALITY_AUDIT.json",
                    out / "data/action_segmentation/train.jsonl",
                    out / "control/COMPLETE",
                    out / "data/vlm/train.jsonl",
                    out / "data/vla/train.json",
                ],
            )
        )
        return steps
    annotations = Path(c.get("annotations", out / "work/generated_annotations.jsonl"))
    steps = []
    if not c.get("annotations"):
        endpoints = [
            x.strip()
            for x in Path(c["endpoint_file"]).read_text().splitlines()
            if x.strip() and not x.lstrip().startswith("#")
        ]
        if not endpoints or len(endpoints) != len(set(endpoints)):
            raise ValueError("endpoint file must contain unique nonempty URLs")
        workers = c.get("workers", len(endpoints) * 16)
        steps.append(
            step(
                "generate",
                command(
                    "generate_cot_dataset.py",
                    "generate",
                    "--input-vlm",
                    Path(c["input_root"]) / "data/vlm/train.jsonl",
                    "--annotations",
                    annotations,
                    "--endpoint-file",
                    c["endpoint_file"],
                    "--model",
                    c.get("model", "Qwen/Qwen3.8-Flash-Next"),
                    "--workers",
                    workers,
                    "--batch-size",
                    c.get("batch_size", workers * 2),
                    "--max-rounds",
                    c.get("max_rounds", 4),
                    "--job-attempts",
                    c.get("job_attempts", 3),
                    "--max-tokens",
                    c.get("max_tokens", 3072),
                    "--timeout",
                    c.get("timeout", 600.0),
                    "--resume",
                ),
                [annotations],
            )
        )
    steps.append(
        step(
            "materialize",
            command(
                "prepare_cot_release.py",
                "--input-root",
                c["input_root"],
                "--output-root",
                out,
                "--annotations",
                annotations,
            ),
            [
                out / "data/vlm/train.jsonl",
                out / "data/vla/train.json",
                out / "data/action_segmentation/train.jsonl",
                out / "work/cot5_annotations.jsonl",
                out / "work/REQUEST_JOURNALS.json",
            ],
        )
    )
    steps.append(
        step(
            "cot_quality",
            command(
                "audit_cot_quality.py",
                "--annotations",
                out / "work/cot5_annotations.jsonl",
                "--vlm",
                out / "data/vlm/train.jsonl",
                "--output",
                out / "COT_QUALITY_AUDIT.json",
                "--expected-rows",
                c["expected_rows"],
            ),
            [out / "COT_QUALITY_AUDIT.json"],
        )
    )
    cmd = command(
        "audit_cot_release.py",
        "--release-root",
        out,
        "--annotations",
        out / "work/cot5_annotations.jsonl",
        "--corrections",
        out / "metadata/ACTION_LABEL_CORRECTIONS.jsonl",
        "--expected-rows",
        c["expected_rows"],
        "--expected-episodes",
        c["expected_episodes"],
        "--allow-preserved-style-allocation",
        "--write-metadata",
    )
    if c.get("require_no_short_same_action", True):
        cmd.append("--require-no-short-same-action")
    if "expected_corrections" in c:
        cmd += ["--expected-corrections", str(c["expected_corrections"])]
    steps.append(
        step(
            "release_audit",
            cmd,
            [
                out / "FINAL_AUDIT.json",
                out / "FINAL_REPORT.json",
                out / "control/COMPLETE",
            ],
        )
    )
    return steps


def fingerprint(c):
    import hashlib

    digest = hashlib.sha256(json.dumps(c, sort_keys=True).encode())
    for root in (ROOT / "atomic_seg", ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    # Input file identity must not silently change across resume.
    sources = [
        Path(c[k])
        for k in ("annotations", "manifest", "endpoint_file", "reference_driver")
        if c.get(k)
    ]
    if c.get("input_root"):
        sources += [
            Path(c["input_root"]) / name
            for name in (
                "data/vlm/train.jsonl",
                "data/vla/train.json",
                "metadata/source_manifest.jsonl",
            )
        ]
        sources += [
            p
            for p in (Path(c["input_root"]) / "metadata").glob("*.json*")
            if p not in sources
        ]
    if c.get("source_root"):
        sources.append(Path(c["source_root"]) / "_state/production_audit.json")
    for path in sources:
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def verify_cot_resume(c, state):
    """Recheck images and request receipts before reusing completed stages."""
    if c["pipeline"] != "cot_data":
        return
    out = Path(c["output_root"])
    completed = state["completed"]
    if "materialize" in completed:
        annotations = out / "work/cot5_annotations.jsonl"
    elif "generate" in completed:
        annotations = out / "work/generated_annotations.jsonl"
    else:
        return
    from atomic_seg.common.artifacts import iter_jsonl
    from atomic_seg.cot_data.evidence import validate_annotation

    source_rows = iter_jsonl(Path(c["input_root"]) / "data/vlm/train.jsonl")
    count = 0
    for annotation in iter_jsonl(annotations):
        row = next(source_rows, None)
        if row is None or annotation["row_index"] != count:
            raise ValueError("Resume annotation/source indices differ")
        validate_annotation(annotation, row, journal_root=annotations.parent)
        count += 1
    if next(source_rows, None) is not None or count != c["expected_rows"]:
        raise ValueError("Resume annotations do not cover the source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = load_config(args.config)
    steps = build_steps(c)
    if args.dry_run:
        print(json.dumps({"config": c, "steps": steps}, ensure_ascii=False, indent=2))
        return
    if c["pipeline"] == "pretrain_noncot":
        # The native workflow owns its per-episode/request caches and writer lock.
        subprocess.run(steps[0]["command"], cwd=ROOT, check=True)
        return
    out = Path(c["output_root"])
    state_path = out / "control/PIPELINE_STATE.json"
    with writer_lock(out / "control/pipeline.lock") as lock_fd:
        identity = fingerprint(c)
        state = {"fingerprint": identity, "completed": {}, "started": utc_now()}
        if state_path.exists():
            if not args.resume:
                raise FileExistsError("Existing pipeline; use --resume")
            state = json.loads(state_path.read_text())
            if state["fingerprint"] != identity:
                raise ValueError(
                    "Inputs, config or code changed; use a new output_root"
                )
            try:
                verify_cot_resume(c, state)
            except Exception:
                (out / "control/COMPLETE").unlink(missing_ok=True)
                raise
        elif any(p.name != "control" for p in out.iterdir()) or any(
            p.name != "pipeline.lock" for p in (out / "control").iterdir()
        ):
            raise FileExistsError(
                "Output contains unrelated files; select a fresh output_root"
            )
        atomic_write_json(state_path, state)
        (out / "logs").mkdir(parents=True, exist_ok=True)
        rerun = False
        for step in steps:
            cached = state["completed"].get(step["name"])
            if (
                not rerun
                and cached
                and all(
                    Path(p).is_file() and sha256_file(Path(p)) == h
                    for p, h in cached.items()
                )
            ):
                print(f"resume: {step['name']} verified", flush=True)
                continue
            rerun = True
            state.pop("finished", None)
            state["completed"].pop(step["name"], None)
            atomic_write_json(state_path, state)
            print(f"running: {step['name']}", flush=True)
            with (out / "logs" / (step["name"] + ".log")).open("a") as log:
                child = subprocess.Popen(
                    step["command"],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock_fd,),
                )

                def stop(signum, frame):
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise KeyboardInterrupt

                previous = {
                    s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)
                }
                try:
                    code = child.wait()
                finally:
                    for s, handler in previous.items():
                        signal.signal(s, handler)
                if code:
                    raise RuntimeError(
                        f"{step['name']} exited {code}; see {out / 'logs' / (step['name'] + '.log')}"
                    )
            state["completed"][step["name"]] = {
                p: sha256_file(Path(p)) for p in step["outputs"]
            }
            if c["pipeline"] == "atomic_actions":
                for path in out.glob("episode_*.json"):
                    state["completed"][step["name"]][str(path)] = sha256_file(path)
            atomic_write_json(state_path, state)
        state["finished"] = utc_now()
        atomic_write_json(state_path, state)
        print(f"Pipeline complete: {out}")


if __name__ == "__main__":
    main()
