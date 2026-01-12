from collections import deque
from datetime import datetime

import pytz
from textual_plotext import PlotextPlot

from sysmon._utils import LOCAL_TIMEZONE, PLOT_HISTORY_SIZE

from ._series import SeriesData, SeriesStyle
from ._types import ValueFormatter, get_default_color


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

    border_title: str
    history_size: int
    marker: str = "braille"
    series: dict[str, SeriesData]
    series_formatters: dict[str, ValueFormatter]
    tz: pytz.BaseTzInfo
    use_temperature_colors: bool
    value_formatter: ValueFormatter | None
    y_upper_lim: float | None

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        tz: str = LOCAL_TIMEZONE,
        history_size: int = PLOT_HISTORY_SIZE,
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

        color_idx = 0
        for series_name, val in value.items():
            if series_name not in self.series:
                color, color_offset = get_default_color(
                    index=color_idx, name=series_name
                )
                self.series[series_name] = SeriesData(
                    values=deque(maxlen=self.history_size),
                    style=SeriesStyle(color=color),
                )
                color_idx += color_offset

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
