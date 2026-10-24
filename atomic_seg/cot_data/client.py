"""Strict multimodal requests with an append-only audit journal."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
from pathlib import Path

from atomic_seg.common.artifacts import compact_json, sha256_file, utc_now
from atomic_seg.common.model_client import (
    DIRECT_OPENER, multimodal_content, normalize_endpoint,
)

from .evidence import digest, normalize_plan_reference, request_payload
from .prompts import PROMPT_VERSION, SYSTEMS


class Journal:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mutex = threading.Lock()
        if self.path.exists():
            with self.path.open("r+b") as stream:
                while raw := stream.readline():
                    if not raw.endswith(b"\n"):
                        backup = self.path.with_name(
                            self.path.name + f".incomplete-{time.time_ns()}"
                        )
                        backup.write_bytes(raw)
                        stream.seek(-len(raw), os.SEEK_CUR)
                        stream.truncate()
                        stream.flush()
                        os.fsync(stream.fileno())
                        break
                    json.loads(raw)

    def append(self, value):
        raw = (compact_json(value) + "\n").encode()
        with self.mutex, self.path.open("ab") as stream:
            offset = stream.tell()
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return {
            "journal": str(self.path), "byte_offset": offset, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }


def parse_response(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Require one model choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Model response is incomplete or truncated")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing model message")
    if message.get("reasoning_content") or message.get("reasoning"):
        raise ValueError("Hidden reasoning returned despite disabled thinking")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Missing explicit assistant content")
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("Assistant must return one strict JSON object")
    return value


class ModelClient:
    def __init__(self, endpoints, *, model, timeout, max_tokens, journal,
                 concurrency=1, transport_attempts=3):
        self.endpoints = [normalize_endpoint(ep).base_url for ep in endpoints]
        if not self.endpoints:
            raise ValueError("No model endpoints")
        self.model, self.timeout, self.max_tokens = model, timeout, max_tokens
        self.journal, self.transport_attempts = journal, transport_attempts
        self.slots = threading.BoundedSemaphore(concurrency)
        self.mutex, self.next_endpoint = threading.Lock(), 0

    def call(self, stage, frozen, payload, *, row_index, attempt):
        attachments = frozen["visible"]["images"]
        paths = [Path(image["path"]) for image in attachments]
        for path, image in zip(paths, attachments):
            if sha256_file(path) != image["sha256"]:
                raise ValueError("Student image changed during generation")
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEMS[stage]},
                {"role": "user", "content": multimodal_content(
                    compact_json(payload), paths,
                    [image["id"] for image in attachments],
                )},
            ],
            "temperature": {"observe": 0.2, "write": 0.4,
                            "grounding_critic": 0.0, "action_critic": 0.0}[stage],
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": False},
        }
        raw = compact_json(body).encode()
        import urllib.request

        for transport in range(1, self.transport_attempts + 1):
            with self.slots:
                with self.mutex:
                    endpoint = self.endpoints[self.next_endpoint % len(self.endpoints)]
                    self.next_endpoint += 1
                started = time.monotonic()
                record = {
                    "stage": stage, "row_index": row_index, "attempt": attempt,
                    "transport_attempt": transport, "endpoint": endpoint,
                    "created_at_utc": utc_now(), "model": self.model,
                    "prompt_version": PROMPT_VERSION, "system_prompt": SYSTEMS[stage],
                    "user_payload": payload, "image_attachments": attachments,
                    "student_visible_sha256": frozen["student_visible_sha256"],
                    "parameters": {k: v for k, v in body.items() if k != "messages"},
                    "payload_sha256": hashlib.sha256(raw).hexdigest(),
                }
                try:
                    request = urllib.request.Request(
                        normalize_endpoint(endpoint).chat_url, data=raw,
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with DIRECT_OPENER.open(request, timeout=self.timeout) as response:
                        record["http_status"] = response.status
                        record["response_text"] = response.read().decode("utf-8")
                    value = parse_response(json.loads(record["response_text"]))
                    record["parsed_content"] = value
                    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    receipt = self.journal.append(record)
                    return value, {**receipt, "endpoint": endpoint}
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, urllib.error.HTTPError):
                        record["error_response"] = exc.read().decode(errors="replace")[:2000]
                    record["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    self.journal.append(record)
                    if not isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)) or transport == self.transport_attempts:
                        raise
            time.sleep(min(8, 2 ** (transport - 1)))


def validate_receipt(receipt, stage, frozen, value, *, body, observer, style,
                     model, row_index, journal_root=None):
    path = Path(receipt["journal"])
    if not path.is_absolute():
        if journal_root is None:
            raise ValueError("Relative request journal requires its annotation directory")
        path = Path(journal_root) / path
    with path.open("rb") as stream:
        stream.seek(receipt["byte_offset"])
        raw = stream.read(receipt["bytes"])
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("Annotation request journal changed")
    record = json.loads(raw)
    if record.get("error") or record["stage"] != stage or record["prompt_version"] != PROMPT_VERSION or record["system_prompt"] != SYSTEMS[stage]:
        raise ValueError("Annotation stage receipt mismatch")
    if record["model"] != model or record["row_index"] != row_index:
        raise ValueError("Annotation model or source index mismatch")
    if record["image_attachments"] != frozen["visible"]["images"] or record["student_visible_sha256"] != frozen["student_visible_sha256"]:
        raise ValueError("Annotation stage image or visible input mismatch")
    parameters = record["parameters"]
    if parameters["chat_template_kwargs"] != {"enable_thinking": False, "preserve_thinking": False}:
        raise ValueError("Teacher hidden thinking must be disabled")
    if parameters["model"] != model or parameters["response_format"] != {"type": "json_object"}:
        raise ValueError("Teacher model or output format mismatch")
    if parameters["temperature"] != {
        "observe": 0.2, "write": 0.4, "grounding_critic": 0.0, "action_critic": 0.0,
    }[stage] or type(parameters["max_tokens"]) is not int or parameters["max_tokens"] < 1:
        raise ValueError("Teacher stage parameters mismatch")
    parsed = parse_response(json.loads(record["response_text"]))
    normalized, _ = normalize_plan_reference(parsed, stage)
    if normalized != value:
        raise ValueError("Annotation differs from the recorded model response")
    payload = record["user_payload"]
    expected = request_payload(
        stage, frozen, observer=observer, reasoning=body, style=style,
        feedback=payload.get("revision_feedback"),
        previous_draft=payload.get("last_candidate_to_revise"),
    )
    if digest(payload) != digest(expected):
        raise ValueError("Annotation teacher evidence isolation mismatch")
    request = {"model": model, "messages": [
        {"role": "system", "content": SYSTEMS[stage]},
        {"role": "user", "content": multimodal_content(
            compact_json(payload),
            [Path(image["path"]) for image in frozen["visible"]["images"]],
            [image["id"] for image in frozen["visible"]["images"]],
        )},
    ], **{k: v for k, v in parameters.items() if k != "model"}}
    if hashlib.sha256(compact_json(request).encode()).hexdigest() != record["payload_sha256"]:
        raise ValueError("Recorded multimodal request hash mismatch")
