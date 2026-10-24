"""Stable numeric formatting shared by all action-space adapters."""

from typing import Any, Sequence


def round_vector(values: Sequence[Any]) -> list[float]:
    result = []
    for value in values:
        rounded = round(float(value), 4)
        result.append(0.0 if rounded == 0.0 else rounded)
    return result
