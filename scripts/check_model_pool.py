#!/usr/bin/env python3
"""Validate endpoint count and model readiness; optional multimodal smoke test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from atomic_seg.common.model_client import (
    DIRECT_OPENER,
    normalize_endpoint,
    request_model,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--expected-endpoints", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.8-Flash-Next")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    endpoints = [
        normalize_endpoint(x.strip())
        for x in args.endpoint_file.read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    if len(endpoints) != args.expected_endpoints or len(
        {e.base_url for e in endpoints}
    ) != len(endpoints):
        raise ValueError("Endpoint count/uniqueness mismatch")
    if not endpoints:
        raise ValueError("Empty pool")

    def check(endpoint):
        try:
            url = endpoint.chat_url.removesuffix("/chat/completions") + "/models"
            with DIRECT_OPENER.open(url, timeout=args.timeout) as response:
                body = json.load(response)
            if args.model not in {m["id"] for m in body["data"]}:
                raise ValueError("Requested model not served")
            if args.image:
                result, _ = request_model(
                    endpoint,
                    model=args.model,
                    system="Return JSON only.",
                    prompt='Describe the image briefly as {"description":"..."}.',
                    image_paths=[args.image],
                    image_roles=["current_head"],
                    timeout=args.timeout,
                    max_tokens=128,
                    temperature=0,
                    seed=1,
                    enable_thinking=False,
                )
                if not result.get("description"):
                    raise ValueError("Empty multimodal response")
            return {"endpoint": endpoint.base_url, "passed": True}
        except Exception as exc:
            return {"endpoint": endpoint.base_url, "passed": False, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=min(32, len(endpoints))) as pool:
        results = list(pool.map(check, endpoints))
    print(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
