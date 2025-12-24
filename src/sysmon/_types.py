import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, StrEnum, auto
from typing import NamedTuple, TypeVar

DEFAULT_HISTORY = 100
LOCAL_TIMEZONE: str = os.getenv("LOCAL_TIMEZONE", default="US/Arizona")

DEFAULT_DMON_POLL = 1
DEFAULT_PROC_POLL = 3
GIB = 1024**3
MIB = 1024**2
KIB = 1024


E = TypeVar("E", bound=Enum)


def enum_from_value(enum: type[E], value: object, /) -> E:
    """Validate and coerce a value into an enum instance.
    Args:
        enum: The enum class to create an instance of.
        value: The value to convert to an enum instance.
    Returns:
        An instance of the enum class.
    Raises:
        ValueError: If the value cannot be converted to an enum instance.
    """
    try:
        if isinstance(value, enum):
            return value
        elif isinstance(value, str):
            str_val: str = value.removeprefix(".").replace("-", "_").strip()
            if str_val in enum:
                return enum(str_val)
            elif str_val.upper() in enum:
                return enum(str_val.upper())
            elif str_val.lower() in enum:
                return enum(str_val.lower())
            try:
                return enum[str_val.upper()]
            except KeyError:
                return enum[str_val.lower()]
        return enum(value)
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"Unsupported {enum.__name__} value: {value!r} - choices "
            f"are: {list(enum.__members__)}"
        ) from None


class TemperatureStatus(StrEnum):
    """Temperature alert status levels."""

    NORMAL = auto()
    WARNING = auto()
    CRITICAL = auto()


class ProcessInfo(NamedTuple):
    """Process information for display in process table."""

    pid: int
    name: str
    user: str
    cpu_percent: float
    sys_mem_mb: float
    gpu_mem: str


class SortColumn(StrEnum):
    """Available columns for sorting process list."""

    PID = auto()
    NAME = auto()
    USER = auto()
    CPU = auto()
    SYS_MEM = auto()
    GPU_MEM = auto()


@dataclass
class ProcessFilter:
    """Filter and sort configuration for process list.

    Attributes:
        query: Fuzzy search query string
        sort_by: Column to sort by
        ascending: Sort direction
    """

    query: str = ""
    sort_by: SortColumn = SortColumn.CPU
    ascending: bool = False

    def matches(self, process: ProcessInfo) -> bool:
        """Check if process matches the filter query using fuzzy matching."""
        if not self.query:
            return True

        query_lower = self.query.lower()
        # Check PID, name, and user for fuzzy match
        searchable = f"{process.pid} {process.name} {process.user}".lower()

        # Simple fuzzy matching: check if all characters appear in order
        query_idx = 0
        for char in searchable:
            if query_idx < len(query_lower) and char == query_lower[query_idx]:
                query_idx += 1
        return query_idx == len(query_lower)

    def sort_key(self, process: ProcessInfo) -> tuple[float | str, ...]:
        """Return sort key for process based on current settings."""
        match self.sort_by:
            case SortColumn.PID:
                return (process.pid,)
            case SortColumn.NAME:
                return (process.name.lower(),)
            case SortColumn.USER:
                return (process.user.lower(),)
            case SortColumn.CPU:
                return (process.cpu_percent,)
            case SortColumn.SYS_MEM:
                return (process.sys_mem_mb,)
            case SortColumn.GPU_MEM:
                # Handle N/A values for sorting
                try:
                    return (float(process.gpu_mem),)
                except ValueError:
                    return (-1.0,) if not self.ascending else (float("inf"),)


ValueFormatter = Callable[[float], str]

# Standard color palette for series
SERIES_COLORS: list[str] = ["cyan", "magenta", "yellow", "green", "blue", "white"]

# Temperature status colors
TEMP_STATUS_COLORS: dict[TemperatureStatus, str] = {
    TemperatureStatus.NORMAL: "green",
    TemperatureStatus.WARNING: "orange",
    TemperatureStatus.CRITICAL: "red",
}
