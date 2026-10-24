"""OpenAI-compatible multimodal transport independent of dataset semantics."""

from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .artifacts import compact_json

DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def multimodal_content(
    text: str, paths: Sequence[Path], roles: Sequence[str]
) -> list[dict[str, Any]]:
    if len(paths) != len(roles):
        raise ValueError("image paths and roles must have equal lengths")
    content = [{"type": "text", "text": text}]
    for role, path in zip(roles, paths):
        content.extend(
            [
                {"type": "text", "text": f"Image role: {role}"},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ]
        )
    return content


def post_chat(
    *,
    endpoint: str,
    payload: Mapping[str, Any],
    timeout: float,
    api_key: str | None = None,
    direct: bool = False,
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        normalize_endpoint(endpoint).chat_url,
        data=compact_json(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    opener = DIRECT_OPENER.open if direct else urllib.request.urlopen
    with opener(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices", [])
    if choices and str(choices[0].get("finish_reason", "")).lower() in {
        "length",
        "max_tokens",
    }:
        raise ValueError("model response was truncated")
    return body


def _data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _extract_json_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        value = None
        for match in re.finditer(r"\{", stripped):
            try:
                candidate, end = decoder.raw_decode(stripped[match.start() :])
            except json.JSONDecodeError:
                continue
            if stripped[match.start() + end :].strip() in {"", "```"}:
                value = candidate
                break
        if value is None:
            raise ValueError("response does not contain one JSON object")
    if not isinstance(value, dict):
        raise ValueError("response JSON must be an object")
    return value


def _response_parts(payload: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("API response has no choices[0]")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise ValueError("API choices[0] is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise ValueError("API response has no message")
    content = message.get("content", "")
    direct_output: dict[str, Any] | None = None
    if isinstance(content, Mapping):
        direct_output = dict(content)
        content = compact_json(direct_output)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, Mapping)
        )
    hidden = message.get("reasoning_content", message.get("reasoning", ""))
    if isinstance(hidden, list):
        hidden = "".join(
            str(item.get("text", "")) for item in hidden if isinstance(item, Mapping)
        )
    hidden_text = "" if hidden is None else str(hidden).strip()
    if direct_output is not None:
        output = direct_output
    else:
        try:
            output = _extract_json_text(str(content))
        except ValueError as content_error:
            if not hidden_text:
                raise ValueError(
                    f"{content_error}; content_tail={str(content)[-500:]!r}"
                ) from content_error
            try:
                output = _extract_json_text(hidden_text)
            except ValueError as hidden_error:
                raise ValueError(
                    f"{content_error}; hidden_error={hidden_error}; "
                    f"content_tail={str(content)[-500:]!r}; "
                    f"hidden_tail={hidden_text[-500:]!r}"
                ) from hidden_error
    return output, hidden_text, str(choice.get("finish_reason", ""))


@dataclass(frozen=True)
class Endpoint:
    base_url: str

    @property
    def chat_url(self) -> str:
        value = self.base_url.rstrip("/")
        if value.endswith("/v1/chat/completions"):
            return value
        if value.endswith("/v1"):
            return value + "/chat/completions"
        return value + "/v1/chat/completions"


def normalize_endpoint(value: str) -> Endpoint:
    raw = value.strip().rstrip("/")
    if "://" not in raw:
        raw = "http://" + raw
    parsed = urllib.parse.urlsplit(raw)
    if not parsed.hostname:
        raise ValueError(f"invalid endpoint: {value}")
    path = parsed.path.rstrip("/")
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
    return Endpoint(
        urllib.parse.urlunsplit(
            (parsed.scheme or "http", parsed.netloc, path, "", "")
        ).rstrip("/")
    )


def non_loopback_endpoints(values: Iterable[str]) -> list[str]:
    """Return unique physical endpoint URLs, excluding per-rank localhost aliases."""

    output: set[str] = set()
    for value in values:
        normalized = normalize_endpoint(str(value)).base_url
        hostname = urllib.parse.urlsplit(normalized).hostname
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            continue
        output.add(normalized)
    return sorted(output)


def request_model(
    endpoint: Endpoint,
    *,
    model: str,
    system: str,
    prompt: str,
    image_paths: Sequence[Path],
    image_roles: Sequence[str],
    timeout: float,
    max_tokens: int,
    temperature: float,
    seed: int,
    enable_thinking: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    content = multimodal_content(prompt, image_paths, image_roles)
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning_effort": "xhigh" if enable_thinking else "medium",
        "seed": seed,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {
            "enable_thinking": bool(enable_thinking),
            "preserve_thinking": False,
        },
    }
    started = time.monotonic()
    try:
        body = post_chat(
            endpoint=endpoint.base_url,
            payload=request_payload,
            timeout=timeout,
            direct=True,
        )
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"HTTP {exc.code} from {endpoint.base_url}: {detail}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise RuntimeError(f"request to {endpoint.base_url} failed: {exc}") from exc
    output, hidden, finish_reason = _response_parts(body)
    if finish_reason.lower() in {"length", "max_tokens"}:
        raise RuntimeError("model response was truncated")
    usage = body.get("usage", {}) if isinstance(body, Mapping) else {}
    metadata = {
        "endpoint": endpoint.base_url,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "usage": usage,
        "hidden_reasoning_present": bool(hidden),
        "hidden_reasoning_words": len(re.findall(r"\b[\w'-]+\b", hidden)),
        "hidden_reasoning_sha256": (
            hashlib.sha256(hidden.encode("utf-8")).hexdigest() if hidden else None
        ),
        "finish_reason": finish_reason,
    }
    return output, metadata
