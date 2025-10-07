from collections import deque
from collections.abc import Callable
from datetime import datetime

import pytz
from textual_plotext import PlotextPlot

from ._defaults import DEFAULT_HISTORY, LOCAL_TIMEZONE

ValueFormatter = Callable[[float], str]


def memory_formatter(total_bytes: float, normed: bool = False) -> ValueFormatter:
    """Formats a memory usage percentage and total bytes into a readable string."""

    def formatter(percent: float, /) -> str:
        normed_percent: float = percent if normed else percent / 100.0
        unnormed_percent: float = percent if not normed else percent * 100.0
        tot_bytes: float = total_bytes * normed_percent
        val: str = f"{unnormed_percent: .1f}% -"
        if tot_bytes >= 1e9:
            val += f"{tot_bytes / 1e9: .2f} GiB"
        elif tot_bytes >= 1e6:
            val += f"{tot_bytes / 1e6: .2f} MiB"
        elif tot_bytes >= 1e3:
            val += f"{tot_bytes / 1e3: .2f} KiB"
        else:
            val += f"{tot_bytes: .2f} B"
        return val

    return formatter


def percent_formatter(normed: bool = True) -> ValueFormatter:
    """Formats a float value as a percentage string with one decimal place."""

    def formatter(percent: float, /) -> str:
        if not normed:
            percent = percent / 100.0
        return f"{percent: .1f}%"

    return formatter


def unit_formatter(unit: str) -> ValueFormatter:
    """Formats a float value as a percentage string with one decimal place."""

    def formatter(value: float, /) -> str:
        return f"{value:.1f} {unit}"

    return formatter


class DataPlot(PlotextPlot):
    """A reactive plotext widget for displaying time-series data."""

    marker: str = "braille"
    data: deque[tuple[datetime, float]]
    tz: pytz.BaseTzInfo
    y_upper_lim: float | None
    value_formatter: ValueFormatter | None

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        tz: str = LOCAL_TIMEZONE,
        history_size: int = DEFAULT_HISTORY,
        value_formatter: ValueFormatter | None = None,
        y_upper_lim: float | None = None,
    ) -> None:
        super().__init__(name=name, id=id)
        self.data = deque(maxlen=history_size)
        self.y_upper_lim = y_upper_lim
        self.value_formatter = value_formatter
        self.tz = pytz.timezone(zone=tz)

    @property
    def formatter_is_set(self) -> bool:
        """Checks if a custom value formatter is set."""
        return self.value_formatter is not None

    def set_value_formatter(self, formatter: ValueFormatter, /) -> None:
        """Sets a custom value formatter for the plot labels."""
        self.value_formatter = formatter

    def set_y_upper_lim(self, upper: float, /) -> None:
        """Sets the upper limit for the y-axis."""
        self.y_upper_lim = upper

    def update_data(self, value: float, /) -> None:
        """Appends the new data point with the current timestamp."""
        self.data.append((datetime.now(self.tz), value))
        self.draw_plot()
        self.refresh()

    def draw_plot(self) -> None:
        """Draws the plot with a time-based x-axis."""
        self.plt.clf()
        self.plt.theme(theme="dark")
        if self.data:
            x_times, y_values = zip(*self.data, strict=False)
            last_val = y_values[-1]
            self.plt.plot(
                y_values,
                marker=self.marker,
                color="cyan",
                label=self.value_formatter(last_val)
                if self.value_formatter
                else f"{last_val:.2f}",
            )
            self.plt.xticks(
                list(range(len(x_times))), labels=[f"{t:%H:%M:%S}" for t in x_times]
            )

            self.plt.ylim(lower=0, upper=self.y_upper_lim)

        else:
            width, height = self.size.width or 60, self.size.height or 15
            self.plt.text(
                "Waiting for data...",
                width // 2,
                height // 2,
                alignment="center",
            )
