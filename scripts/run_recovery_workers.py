#!/usr/bin/env python3
"""Run disjoint native-recovery data ranks against a shared endpoint pool."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import os
import signal
import subprocess
import time

from atomic_seg.common.locking import writer_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--reference-driver", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--generation-workers", type=int, default=16)
    parser.add_argument("--max-rank-attempts", type=int, default=3)
    args = parser.parse_args()
    if (
        min(
            args.world_size,
            args.prepare_workers,
            args.generation_workers,
            args.max_rank_attempts,
        )
        < 1
    ):
        parser.error("worker and rank counts must be positive")
    children, logs, commands, attempts = [], [], [], []
    with writer_lock(args.output_root / "control/rank_pool.lock") as lock_fd:

        def stop(signum=None, frame=None):
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            if signum is not None:
                raise KeyboardInterrupt

        previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
        try:
            log_dir = args.output_root / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            for rank in range(args.world_size):
                log = (log_dir / f"rank_{rank:02d}.log").open("a")
                logs.append(log)
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("recovery_cot.py")),
                    "run-rank",
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(args.world_size),
                    "--output-root",
                    str(args.output_root),
                    "--endpoint-file",
                    str(args.endpoint_file),
                    "--reference-driver",
                    str(args.reference_driver),
                    "--model",
                    args.model,
                    "--prepare-workers",
                    str(args.prepare_workers),
                    "--generation-workers",
                    str(args.generation_workers),
                    "--resume",
                    "--launch-attempt",
                    "1",
                ]
                commands.append(command)
                attempts.append(1)
                children.append(
                    subprocess.Popen(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        pass_fds=(lock_fd,),
                    )
                )
            while True:
                failed = [
                    i
                    for i, child in enumerate(children)
                    if child.poll() not in (None, 0)
                ]
                if failed:
                    exhausted = [
                        i for i in failed if attempts[i] >= args.max_rank_attempts
                    ]
                    if exhausted:
                        raise RuntimeError(
                            f"Recovery ranks exhausted retries: {exhausted}; see {log_dir}"
                        )
                    for i in failed:
                        attempts[i] += 1
                        commands[i][-1] = str(attempts[i])
                        children[i] = subprocess.Popen(
                            commands[i],
                            stdout=logs[i],
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            pass_fds=(lock_fd,),
                        )
                if all(child.poll() == 0 for child in children):
                    break
                time.sleep(1)
            if any(child.returncode for child in children):
                raise RuntimeError("Recovery rank failed")
        finally:
            stop()
            for s, handler in previous.items():
                signal.signal(s, handler)
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
