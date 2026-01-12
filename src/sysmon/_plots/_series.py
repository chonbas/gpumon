from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from sysmon._utils import PLOT_HISTORY_SIZE

from ._types import TEMP_STATUS_COLORS, TemperatureStatus


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
        default_factory=lambda: deque(maxlen=PLOT_HISTORY_SIZE)
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
            self.style.status_color = self.style.color
        else:
            self.style.status_color = TEMP_STATUS_COLORS.get(status)
