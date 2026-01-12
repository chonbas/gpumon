from sysmon._utils import GIB, KIB, MIB

from ._types import ValueFormatter


def memory_formatter(total_bytes: float, from_percent: bool = False) -> ValueFormatter:
    """Formats a memory usage percentage and total bytes into a readable string."""

    def formatter(value: float, /) -> str:
        percent_value: float = value if from_percent else value / total_bytes * 100.0
        bytes_value: float = (
            value if not from_percent else total_bytes * (percent_value / 100.0)
        )

        val: str = f"{percent_value: .1f}% ("
        if bytes_value >= GIB:
            val += f"{bytes_value / GIB:.1f}GB"
        elif bytes_value >= MIB:
            val += f"{bytes_value / MIB:.1f}MB"
        elif bytes_value >= KIB:
            val += f"{bytes_value / KIB:.1f}KB"
        else:
            val += f"{bytes_value:.1f}B"
        val += f"/{total_bytes / GIB:.1f}GB)"
        return val.strip()

    return formatter


def percent_formatter(normed: bool = True) -> ValueFormatter:
    """Formats a float value as a percentage string with one decimal place."""

    def formatter(percent: float, /) -> str:
        if not normed:
            percent = percent / 100.0
        return f"{percent:.1f}%"

    return formatter


def unit_formatter(unit: str) -> ValueFormatter:
    """Formats a float value with one decimal place."""

    def formatter(value: float, /) -> str:
        return f"{value:.1f}{unit}"

    return formatter
