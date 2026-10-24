#!/usr/bin/env python3
"""Single-host AgiBot API production, preflight and durable resume."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def freeze(run):
    from atomic_seg.pretrain.realomni.storage import (
        atomic_json, commit_directory, digest, file_hash, lock, read_json, staging, verify_directory,
    )
    target = run / "source"
    with lock(run / "control/source.lock", blocking=True):
        if target.exists():
            manifest = read_json(target / "FROZEN.json")
            verify_directory(target, digest(manifest))
            return target
        temporary = staging(run, "source")
        shutil.copytree(ROOT / "atomic_seg", temporary / "atomic_seg",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (temporary / "scripts").mkdir()
        shutil.copyfile(Path(__file__), temporary / "scripts/agibot.py")
        font = ROOT / "scripts/assets/agibot-review-cjk.woff"
        if font.is_file():
            (temporary / "scripts/assets").mkdir()
            shutil.copyfile(font, temporary / "scripts/assets" / font.name)
        files = {str(path.relative_to(temporary)): file_hash(path)
                 for path in sorted(temporary.rglob("*")) if path.is_file()}
        manifest = {"files": files, "source_workspace": str(ROOT)}
        atomic_json(temporary / "FROZEN.json", manifest)
        commit_directory(temporary, target, digest(manifest))
    return target


def checked_settings(args):
    from atomic_seg.pretrain.agibot.config import Settings

    settings = Settings(source_root=str(Path(args.source).resolve()), model=args.model,
                        reasoning_effort=args.reasoning_effort, temperature=args.temperature,
                        max_tokens=args.max_tokens, batch_size=args.batch_size,
                        object_grounding=args.object_grounding)
    source, run = Path(settings.source_root), args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dataset directory is not accessible: {source}")
    if run == source or run in source.parents or source in run.parents:
        raise ValueError("source and output must be separate, non-overlapping directories")
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if args.progress_every < 1:
        raise ValueError("progress interval must be positive")
    selection = None
    if args.selection:
        selection = json.loads(args.selection.read_text())
        if not isinstance(selection, list) or not selection or any(not isinstance(x, str) for x in selection):
            raise ValueError("selection must be a nonempty JSON list of episode identities")
        if len(set(selection)) != len(selection):
            raise ValueError("selection contains duplicate episode identities")
    config_path = run / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        if config["settings"] != json.loads(json.dumps(settings.as_dict())):
            raise ValueError("semantic configuration changed; use a new AGIBOT_OUTPUT directory")
        if config["selection"] != (sorted(selection) if selection else None):
            raise ValueError("episode selection changed; use a new AGIBOT_OUTPUT directory")
    return settings, selection


def main():
    from atomic_seg.pretrain.agibot.config import API_BASE, BASE_POLICY, MODEL, SOURCE_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="start",
                        choices=["start", "check", "probe", "prepare", "status", "review"])
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_ROOT) / "run_glm53_flash_xhigh_grounded_v7_16d")
    parser.add_argument("--source", default=SOURCE_ROOT)
    parser.add_argument("--selection", type=Path, help="JSON list of exact task/episode_000000 identities; omitted means ALL")
    parser.add_argument("--api-base", default=API_BASE)
    parser.add_argument("--api-token-file", type=Path, default=ROOT / ".api_token")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh", "max"], default="xhigh")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--object-grounding", action="store_true",
                        help="Use an image-first object ledger before sub_task annotation")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print a progress summary every N processed episodes per pass (default: 1000)")
    parser.add_argument("--stage", choices=["inspect", "segment", "media", "all"], default="all")
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()

    config_path = args.output / "run_config.json"
    if args.command in {"start", "prepare"} and config_path.is_file():
        config = json.loads(config_path.read_text())
        if config["settings"].get("base_policy") != BASE_POLICY:
            raise ValueError(
                "output uses the old chassis filtering policy; set a new AGIBOT_OUTPUT "
                "directory for the 16D dual-arm policy (for example, ending in _16d_split2)"
            )

    if args.command in {"start", "prepare", "status", "review"}:
        source = args.output / "source"
        if source.is_dir() and ROOT != source:
            source = freeze(args.output)
            os.execv(sys.executable, [sys.executable, str(source / "scripts/agibot.py"), *sys.argv[1:]])

    from atomic_seg.pretrain.agibot.runner import initialize, implementation, work, summarize
    if args.command == "status":
        print(json.dumps(summarize(args.output), indent=2, ensure_ascii=False))
        return
    if args.command == "review":
        from atomic_seg.pretrain.agibot.review import build
        print(build(args.output, args.preview_output or args.output / "preview"))
        return

    settings, selection = checked_settings(args)
    implementation()
    token = None
    if args.command in {"check", "probe"} or (args.command == "start" and args.stage == "all"):
        from atomic_seg.pretrain.agibot.api import check, load_token, probe
        token = load_token(args.api_token_file)
        print(json.dumps(check(args.api_base, token, settings), ensure_ascii=False), flush=True)
        if args.command == "probe" or (args.command == "start" and ROOT == args.output / "source"):
            print(json.dumps(probe(args.api_base, token, settings), ensure_ascii=False), flush=True)
    if args.command in {"check", "probe"}:
        return
    if ROOT != args.output / "source":
        source = freeze(args.output)
        os.execv(sys.executable, [sys.executable, str(source / "scripts/agibot.py"), *sys.argv[1:]])
    version = initialize(args.output, settings, selection)
    print(json.dumps({"version": version, "output": str(args.output),
                      "settings": settings.as_dict(), "concurrency": args.concurrency}), flush=True)
    if args.command == "prepare":
        return
    print(json.dumps(work(args.output, settings, [args.api_base], args.stage,
                          concurrency=args.concurrency, api_key=token,
                          progress_every=args.progress_every), ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun agibot.sh to resume committed stages.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        print(f"agibot: {error}", file=sys.stderr)
        raise SystemExit(2)
