
from __future__ import annotations
from typing import Iterable

TIMESTEP_OPTIONS: list[tuple[str, int | None]] = [
    ("auto", None),
    ("none", None),
    ("1 second", 1),
    ("1 minute", 60),
    ("5 minutes", 300),
    ("15 minutes", 900),
    ("1 hour", 3600),
    ("6 hours", 21600),
    ("12 hours", 43200),
    ("daily", 86400),
    ("weekly", 604800),
]

MOVING_AVERAGE_OPTIONS: list[tuple[str, int | None]] = list(TIMESTEP_OPTIONS)

TIMESTEP_SECONDS: list[int] = [seconds for _, seconds in TIMESTEP_OPTIONS if seconds]


def label_to_seconds(label: str, options: Iterable[tuple[str, int | None]]) -> int | None:
    for option_label, seconds in options:
        if option_label == label:
            return seconds
    return None


def seconds_to_label(seconds: int, options: Iterable[tuple[str, int | None]]) -> str | None:
    for option_label, option_seconds in options:
        if option_seconds == seconds:
            return option_label
    return None
