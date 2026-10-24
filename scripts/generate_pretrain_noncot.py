#!/usr/bin/env python3
"""Prepare and review OXE/ABC native non-CoT data using dataset-specific prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_seg.common.artifacts import atomic_write_json
from atomic_seg.common.locking import writer_lock
from atomic_seg.pretrain.pipeline import annotate, export, load_prepared, prepare


def load_config(path):
    config = json.loads(path.read_text())
    if config.get("pipeline") != "pretrain_noncot" or not config.get("datasets"):
        raise ValueError("pretrain_noncot requires a nonempty datasets list")
    for key in ["workers", "batch_size"]:
        value = config.get(key, 4)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not config.get("model"):
        raise ValueError("model is required")
    for key in ["output_root", "endpoint_file"]:
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve())
    for key in ("reuse_prepared_root", "reuse_annotations_root"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
            reused, out = Path(config[key]), Path(config["output_root"])
            if not reused.is_dir() or reused == out or reused in out.parents or out in reused.parents:
                raise ValueError(f"{key} must be a separate existing output")
    endpoint_path = Path(config["endpoint_file"])
    if not endpoint_path.is_file() or not any(x.strip() and not x.lstrip().startswith("#") for x in endpoint_path.read_text().splitlines()):
        raise ValueError("endpoint_file must list at least one endpoint")
    ids = []
    for dataset in config["datasets"]:
        if dataset["adapter"] not in {"oxe_bridge", "abc_yam"}:
            raise ValueError("unsupported dataset profile")
        if dataset.get("length_policy") is not None:
            from atomic_seg.pretrain.length_policy import LengthPolicy

            if dataset["adapter"] != "abc_yam":
                raise ValueError("length_policy currently supports ABC/YAM only")
            LengthPolicy(**dataset["length_policy"])
        dataset["root"] = str((path.parent / dataset["root"]).resolve())
        root, out = Path(dataset["root"]), Path(config["output_root"])
        if not root.is_dir() or root == out or root in out.parents or out in root.parents:
            raise ValueError("source root must exist and be separate from output")
        ids.extend(c["id"] for c in dataset["episodes"])
    if len(set(ids)) != len(ids) or any(not x.replace("_", "").isalnum() for x in ids):
        raise ValueError("episode ids must be unique safe path components")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=["prepare", "annotate", "export", "review", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if args.dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    output = Path(config["output_root"])
    with writer_lock(output / "control" / "pretrain.lock"):
        saved = output / "run_config.json"
        if saved.exists() and json.loads(saved.read_text()) != config:
            raise ValueError("config changed; select a new output_root")
        atomic_write_json(saved, config)
        episodes = prepare(config) if args.stage in {"prepare", "all"} else load_prepared(config)
        if args.stage in {"annotate", "all"}:
            annotate(config, episodes)
        if args.stage in {"export", "all"}:
            report = export(config, episodes)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.stage in {"review", "all"}:
            from atomic_seg.pretrain.review import build_review

            build_review(config, episodes)


if __name__ == "__main__":
    main()
