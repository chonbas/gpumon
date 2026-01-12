from collections.abc import Callable
from enum import Enum, StrEnum, auto
from typing import TypeVar

ValueFormatter = Callable[[float], str]

E = TypeVar("E", bound=Enum)


class TemperatureStatus(StrEnum):
    """Temperature alert status levels."""

    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()


SERIES_COLORS: list[str] = ["blue", "magenta", "yellow", "green", "white"]
SERIES_NAME_COLOR_MAP: dict[str, str] = {"cpu": "cyan", "system": "cyan"}
TEMP_STATUS_COLORS: dict[TemperatureStatus, str] = {
    TemperatureStatus.NORMAL: "green",
    TemperatureStatus.WARNING: "orange",
    TemperatureStatus.CRITICAL: "red",
}


def get_default_color(index: int, name: str | None = None) -> tuple[str, int]:
    """Get a default color from the standard palette.

    Args:
        name: Optional series name to derive color from
        index: Optional index to select color
    Returns:
        A color string from the SERIES_COLORS palette
    """
    l_name = name.lower() if name else ""
    if name and l_name in SERIES_NAME_COLOR_MAP:
        return SERIES_NAME_COLOR_MAP[l_name], 0
    return SERIES_COLORS[index % len(SERIES_COLORS)], 1
