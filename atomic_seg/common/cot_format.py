"""Shared response envelope; dataset-specific semantics live with each pipeline."""

import json
import re

THINK_RE = re.compile(r"^<think>\n(.*)\n</think>\n\n(\{.*\})$", re.DOTALL)


def word_count(value: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", value))


def split_assistant(content: str) -> tuple[str, str]:
    match = re.fullmatch(
        r"\s*<think>\s*(.*?)\s*</think>\s*(\{.*\})\s*", content, re.DOTALL
    )
    if match is None:
        raise ValueError("assistant content is not <think>...</think> plus JSON")
    reasoning = re.sub(r"\s+", " ", match.group(1)).strip()
    answer_text = match.group(2).strip()
    if not isinstance(json.loads(answer_text), dict):
        raise ValueError("assistant final answer is not a JSON object")
    return reasoning, answer_text


def assistant_with_reasoning(reasoning: str, answer_text: str) -> str:
    return f"<think>\n{reasoning.strip()}\n</think>\n\n{answer_text}"
