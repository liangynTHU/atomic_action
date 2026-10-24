"""Authenticated Chat Completions; credentials never enter run artifacts."""

import io
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

from atomic_seg.common.model_client import DIRECT_OPENER, normalize_endpoint


class ServiceUnavailable(RuntimeError):
    pass


def load_token(path):
    path = Path(path)
    try:
        token = path.read_text().strip()
    except OSError as error:
        raise ServiceUnavailable(f"cannot read API token file: {path}") from error
    if not token or any(character.isspace() for character in token):
        raise ServiceUnavailable("API token file must contain exactly one token")
    return token


def chat_url(endpoint):
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("API base must be an HTTP(S) URL without embedded credentials or query")
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("remote API credentials require HTTPS; use an SSH tunnel for a local service")
    return normalize_endpoint(endpoint).chat_url


def request_json(url, token, *, payload=None, timeout=300):
    headers = {"Accept": "application/json", "Authorization": f"Bearer {token}"}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
    request = urllib.request.Request(url, data=data, headers=headers)
    host = urllib.parse.urlsplit(url).hostname
    opener = DIRECT_OPENER.open if host in {"localhost", "127.0.0.1", "::1"} else urllib.request.urlopen
    try:
        with opener(request, timeout=timeout) as result:
            response = json.load(result)
    except urllib.error.HTTPError as error:
        code = error.code
        error.close()
        hint = {
            401: "check the TT Switch inbound token and upstream CodeBuddy credentials",
            403: "check API token permissions and upstream credentials",
            404: "check the API base and exact model ID",
            429: "API rate limit reached",
        }.get(code, "check the TT Switch service and upstream configuration")
        raise ServiceUnavailable(f"HTTP {code}: {hint}") from None
    except (OSError, urllib.error.URLError, TimeoutError) as error:
        raise ServiceUnavailable(f"API connection failed ({type(error).__name__})") from None
    except (ValueError, UnicodeError):
        raise ServiceUnavailable("API did not return valid JSON") from None
    if not isinstance(response, dict) or response.get("error"):
        raise ServiceUnavailable("API returned an error or a non-object response")
    return response


def generation_parameters(settings):
    return {
        "model": settings.model,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "reasoning_effort": settings.reasoning_effort,
        "response_format": {"type": "json_object"},
    }


def final_json(response):
    try:
        choice = response["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("teacher did not finish normally; output may be truncated")
        content = choice["message"]["content"]
        if not isinstance(content, str):
            raise ValueError("assistant content must be a JSON string")
        content = content.strip()
        if content.startswith("```"):
            lines = content.splitlines()
            if len(lines) < 3 or lines[0] not in {"```json", "```"} or lines[-1] != "```":
                raise ValueError("return one complete JSON object with no surrounding text")
            content = "\n".join(lines[1:-1])
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("teacher JSON must be an object")
        return value
    except (KeyError, TypeError, IndexError, AttributeError):
        raise ValueError("API response has no complete assistant content") from None


def response_audit(response):
    """Keep final output and usage, but never persist private reasoning text."""
    choices = []
    raw_choices = response.get("choices", [])
    for choice in raw_choices if isinstance(raw_choices, list) else []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {})
        if not isinstance(message, dict):
            message = {}
        choices.append({
            "finish_reason": choice.get("finish_reason"),
            "message": {"role": message.get("role", "assistant"), "content": message.get("content")},
            "hidden_reasoning_present": bool(message.get("reasoning") or message.get("reasoning_content")),
        })
    return {"id": response.get("id"), "model": response.get("model"),
            "usage": response.get("usage", {}), "choices": choices}


def check(endpoint, token, settings):
    url = chat_url(endpoint).removesuffix("/chat/completions") + "/models"
    catalog = request_json(url, token, timeout=30)
    models = catalog.get("data", catalog.get("models"))
    if not isinstance(models, list):
        raise ServiceUnavailable("models endpoint returned no model catalog")
    entry = next((item for item in models if isinstance(item, dict) and item.get("id") == settings.model), None)
    if entry is None:
        raise ServiceUnavailable(f"model {settings.model} is absent from the authenticated API catalog")
    if "input" in entry and "image" not in entry["input"]:
        raise ServiceUnavailable("selected model catalog does not declare image support")
    efforts = entry.get("reasoning", {}).get("efforts")
    if efforts and settings.reasoning_effort not in efforts:
        raise ServiceUnavailable("selected model does not support the requested reasoning effort")
    return {"model": settings.model, "api_base": endpoint, "reasoning_effort": settings.reasoning_effort,
            "catalog_accessible": True}


def probe(endpoint, token, settings):
    """One small paid multimodal request before the user starts a full run."""
    import base64
    from PIL import Image

    image = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 0, 0)).save(image, format="PNG")
    payload = {**generation_parameters(settings), "messages": [
        {"role": "system", "content": 'Return only a JSON object with one key, "color".'},
        {"role": "user", "content": [
            {"type": "text", "text": "What color is the image? Use a lowercase English color name."},
            {"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(image.getvalue()).decode("ascii")}},
        ]},
    ]}
    begun = time.monotonic()
    response = request_json(chat_url(endpoint), token, payload=payload)
    if final_json(response) != {"color": "red"}:
        raise ServiceUnavailable("multimodal JSON probe failed; production was not started")
    return {"passed": True, "elapsed_s": time.monotonic() - begun, "response": response_audit(response)}
