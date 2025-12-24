from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import pytz
from textual_plotext import PlotextPlot

from sysmon._types import (
    DEFAULT_HISTORY,
    GIB,
    KIB,
    LOCAL_TIMEZONE,
    MIB,
    SERIES_COLORS,
    TEMP_STATUS_COLORS,
    TemperatureStatus,
    ValueFormatter,
)


@dataclass
class SeriesStyle:
    """Style configuration for a data series.

    Attributes:
        color: Base color for this series
        status_color: Optional override color based on status (e.g., temperature)
    """

    color: str = "cyan"
    status_color: str | None = None

    @property
    def active_color(self) -> str:
        """Return the color to use for rendering."""
        return self.status_color if self.status_color else self.color


@dataclass
class SeriesData:
    """Data and styling for a single series in a plot.

    Attributes:
        values: Deque of (timestamp, value) tuples
        style: Rendering style for this series
        high_threshold: Optional high threshold for warning status
        critical_threshold: Optional critical threshold for alert status
    """

    values: deque[tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=DEFAULT_HISTORY)
    )
    style: SeriesStyle = field(default_factory=SeriesStyle)
    high_threshold: float | None = None
    critical_threshold: float | None = None

    def get_status(self) -> TemperatureStatus:
        """Get current status based on last value and thresholds."""
        if not self.values:
            return TemperatureStatus.NORMAL
        current = self.values[-1][1]
        if self.critical_threshold is not None and current >= self.critical_threshold:
            return TemperatureStatus.CRITICAL
        if self.high_threshold is not None and current >= self.high_threshold:
            return TemperatureStatus.WARNING
        return TemperatureStatus.NORMAL

    def update_status_color(self) -> None:
        """Update the status color based on current value and thresholds.

        Only sets status_color for WARNING or CRITICAL states.
        NORMAL state keeps status_color as None so base color is used.
        """
        status = self.get_status()
        if status == TemperatureStatus.NORMAL:
            # Use base color for normal temps to distinguish series
            self.style.status_color = None
        else:
            self.style.status_color = TEMP_STATUS_COLORS.get(status)


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


class DataPlot(PlotextPlot):
    """A reactive plotext widget for displaying time-series data.

    Supports multiple series with individual styling and temperature-based
    color coding for threshold warnings.

    Attributes:
        marker: Plot marker style (default: braille)
        series: Dictionary mapping series names to their data and style
        tz: Timezone for timestamp display
        y_upper_lim: Optional upper limit for y-axis
        value_formatter: Default formatter for value display
        series_formatters: Per-series formatters (override default)
        history_size: Number of data points to retain per series
        use_temperature_colors: Whether to apply temperature-based coloring
    """

    marker: str = "braille"
    series: dict[str, SeriesData]
    tz: pytz.BaseTzInfo
    y_upper_lim: float | None
    value_formatter: ValueFormatter | None
    series_formatters: dict[str, ValueFormatter]
    history_size: int
    use_temperature_colors: bool

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        tz: str = LOCAL_TIMEZONE,
        history_size: int = DEFAULT_HISTORY,
        value_formatter: ValueFormatter | None = None,
        y_upper_lim: float | None = None,
        use_temperature_colors: bool = False,
    ) -> None:
        super().__init__(name=name, id=id)
        self.series = {}
        self.series_formatters = {}
        self.history_size = history_size
        self.y_upper_lim = y_upper_lim
        self.value_formatter = value_formatter
        self.tz = pytz.timezone(zone=tz)
        self.border_title = name if name else "Data Plot"
        self.use_temperature_colors = use_temperature_colors

    def formatter_is_set(self, series: str = "default") -> bool:
        """Checks if a formatter is set for the given series or as default."""
        return series in self.series_formatters or self.value_formatter is not None

    def set_value_formatter(
        self, formatter: ValueFormatter, /, series: str | None = None
    ) -> None:
        """Sets a value formatter.

        Args:
            formatter: The formatter function
            series: If provided, set formatter for specific series only.
                    If None, set as default formatter for all series.
        """
        if series is None:
            self.value_formatter = formatter
        else:
            self.series_formatters[series] = formatter

    def get_formatter(self, series: str) -> ValueFormatter | None:
        """Get the formatter for a series (per-series or default)."""
        return self.series_formatters.get(series, self.value_formatter)

    def set_y_upper_lim(self, upper: float, /) -> None:
        """Sets the upper limit for the y-axis."""
        self.y_upper_lim = upper

    def set_series_thresholds(
        self,
        series_name: str,
        high: float | None = None,
        critical: float | None = None,
    ) -> None:
        """Set temperature thresholds for a series.

        Args:
            series_name: Name of the series to configure
            high: High threshold for warning status
            critical: Critical threshold for alert status
        """
        if series_name in self.series:
            self.series[series_name].high_threshold = high
            self.series[series_name].critical_threshold = critical

    def update_data(
        self,
        value: float | dict[str, float],
        /,
        thresholds: dict[str, tuple[float | None, float | None]] | None = None,
    ) -> None:
        """Appends the new data point with the current timestamp.

        Args:
            value: Single value or dict mapping series names to values
            thresholds: Optional dict of series names to (high, critical)
        """
        now = datetime.now(self.tz)
        if isinstance(value, (int, float)):
            value = {"default": float(value)}

        color_idx = len(self.series)
        for series_name, val in value.items():
            if series_name not in self.series:
                color = SERIES_COLORS[color_idx % len(SERIES_COLORS)]
                self.series[series_name] = SeriesData(
                    values=deque(maxlen=self.history_size),
                    style=SeriesStyle(color=color),
                )
                color_idx += 1

            series = self.series[series_name]
            series.values.append((now, val))

            # Update thresholds if provided
            if thresholds and series_name in thresholds:
                high, critical = thresholds[series_name]
                series.high_threshold = high
                series.critical_threshold = critical

            # Update status color for temperature plots
            if self.use_temperature_colors:
                series.update_status_color()

        self.draw_plot()
        self.refresh()

    def draw_plot(self) -> None:
        """Draws the plot with a time-based x-axis."""
        self.plt.clf()

        if self.series:
            first_series = True

            for series_name, series_data in self.series.items():
                if not series_data.values:
                    continue
                x_times, y_values = zip(*series_data.values, strict=False)
                last_val = y_values[-1]
                if self.y_upper_lim is not None and last_val > self.y_upper_lim:
                    self.y_upper_lim = last_val * 1.1

                label = ""
                if series_name != "default":
                    label = f"{series_name}: "

                # Use per-series formatter if available, otherwise default
                formatter = self.get_formatter(series_name)
                if formatter:
                    label += formatter(last_val)
                else:
                    label += f"{last_val:.2f}"

                # Use status color for temperature plots, otherwise base color
                color = series_data.style.active_color
                self.plt.plot(y_values, marker=self.marker, color=color, label=label)

                if first_series:
                    self.plt.xticks(
                        list(range(len(x_times))),
                        labels=[f"{t:%H:%M:%S}" for t in x_times],
                    )
                    first_series = False

            self.plt.ylim(lower=0, upper=self.y_upper_lim)

        else:
            width, height = self.size.width or 60, self.size.height or 15
            self.plt.text(
                "Waiting for data...",
                width // 2,
                height // 2,
                alignment="center",
            )
