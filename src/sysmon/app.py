from functools import partial
from math import ceil
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid
from textual.widgets import Footer, Header, RichLog, Static

from sysmon._info_panel import update_info_panel
from sysmon._plots import (
    DataPlot,
    percent_formatter,
    poll_cpu_percent,
    poll_cpu_sys_memory,
    poll_cpu_temp,
    poll_nvidia_dmon_info,
    unit_formatter,
)
from sysmon._processes import ProcessTable
from sysmon._utils import POLL_INTERVAL


class SystemMonitor(App):
    """Monitors NVIDIA GPU statistics, as well as CPU and Memory usage
    in real-time using `nvidia-smi` and `psutil`.

    Features collapsed graphs for memory, utilization, and temperature with
    legends to distinguish between CPU and multiple GPUs.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="l", action="toggle_log", description="Toggle Log"),
        Binding(key="1", action="toggle_plot(0)", description="Memory", show=True),
        Binding(key="2", action="toggle_plot(1)", description="Utilization", show=True),
        Binding(key="3", action="toggle_plot(2)", description="Temperature", show=True),
        Binding(key="4", action="toggle_plot(3)", description="GPU Power", show=True),
        Binding(key="k", action="kill_process", description="Kill Process"),
        Binding(key="f", action="focus_filter", description="Filter"),
    ]
    TITLE = "SystemMonitor"

    CSS = """
    #info-panel {
        height: auto;
        padding: 0 1;
        color: $primary;
        border: round $primary;
        margin-bottom: 1;
        text-align: left;
        text-style: bold;
    }
    #process-table {
        height: 25%;
        width: 1fr;
    }
    #raw-log {
        height: 10%;
    }
    DataPlot, #raw-log {
        border: round $primary;
    }
    DataPlot:disabled {
        opacity: 0.0;
    }

    """
    info_panel: Static
    raw_log: RichLog
    plots: list[DataPlot]
    process_table: ProcessTable

    def __init__(self) -> None:
        super().__init__()
        self.info_panel = Static(content="Querying system info...", id="info-panel")
        # Unified Memory plot (System RAM + GPU VRAM)
        unified_memory = DataPlot(
            name="Memory Usage (%)",
            id="mem-plot",
            y_upper_lim=100,
            # Formatters set per-series by workers based on actual totals
        )

        # Unified Utilization plot (CPU + GPU)
        unified_utilization = DataPlot(
            name="Utilization (%)",
            id="util-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )

        # Unified Temperature plot (CPU + GPU) with color-coded thresholds
        unified_temperature = DataPlot(
            name="Temperature (°C)",
            id="temp-plot",
            value_formatter=unit_formatter(unit="°C"),
            y_upper_lim=50,
            use_temperature_colors=True,
        )

        # GPU Power (separate as it doesn't apply to CPU)
        gpu_power = DataPlot(
            name="GPU Power (W)",
            id="gpu-power-plot",
            value_formatter=unit_formatter(unit="W"),
        )

        self.plots = [
            unified_memory,
            unified_utilization,
            unified_temperature,
            gpu_power,
        ]

        self.process_table = ProcessTable(id="process-table")
        self.raw_log = RichLog(max_lines=10, id="raw-log", highlight=True, markup=True)
        self.raw_log.border_title = "Log (L to Toggle)"
        self.raw_log.display = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.info_panel
        with Grid(id="gpu-grid"):
            for plot in self.plots:
                yield plot
        yield self.process_table
        yield self.raw_log
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "gruvbox"
        self._update_plot_grid_layout()

        unified_memory, unified_utilization, unified_temperature, gpu_power = self.plots

        # Initialize workers for collapsed unified plots
        self.run_worker(
            work=partial(
                update_info_panel,
                self.info_panel,
                self.raw_log,
                unified_memory,
                gpu_power,
            ),
            exclusive=True,
            group="initialization",
        )

        # CPU utilization feeds into unified utilization plot
        self.run_worker(
            work=partial(poll_cpu_percent, self.raw_log, unified_utilization),
            exclusive=True,
            group="cpu_polling",
        )

        # CPU temperature feeds into unified temperature plot with thresholds
        self.run_worker(
            work=partial(poll_cpu_temp, self.raw_log, unified_temperature),
            exclusive=True,
            group="cpu_temp_polling",
        )

        # System memory feeds into unified memory plot
        self.run_worker(
            work=partial(poll_cpu_sys_memory, self.raw_log, unified_memory),
            exclusive=True,
            group="memory_polling",
        )

        # GPU stats feed into unified plots
        self.run_worker(
            work=partial(
                poll_nvidia_dmon_info,
                self.raw_log,
                unified_memory,
                gpu_power,
                unified_temperature,
                unified_utilization,
            ),
            exclusive=True,
            group="dmon_polling",
        )

        self.set_interval(
            callback=self._refresh_process_list,
            interval=POLL_INTERVAL,
        )

    ###################################################################################
    # Keybind hooks
    ###################################################################################
    def action_focus_filter(self) -> None:
        """Focus the process filter input."""
        self.process_table.focus_filter()

    def action_kill_process(self) -> None:
        """Kills the selected process."""
        self.process_table.kill_process(self.raw_log)

    def action_toggle_log(self) -> None:
        self.raw_log.display = not self.raw_log.display

    def action_toggle_plot(self, plot_index: int) -> None:
        """Toggles the display of a plot and updates the grid layout."""
        if 0 <= plot_index < len(self.plots):
            plot = self.plots[plot_index]
            plot.display = not plot.display
            self._update_plot_grid_layout()

    ###################################################################################
    # Private Methods
    ###################################################################################
    async def _refresh_process_list(self) -> None:
        """Wrapper to refresh process list with current filter state."""
        await self.process_table.update_process_list(self.raw_log)

    def _update_plot_grid_layout(self) -> None:
        """Calculates and applies the optimal grid layout based on visible plots."""
        grid: Grid = self.query_one("#gpu-grid", Grid)

        visible_plots: list[DataPlot] = [p for p in self.plots if p.display]
        count: int = len(visible_plots)

        grid.display = bool(count)
        if not count:
            return

        if count <= 2:
            cols: int = count
            rows: int = 1
        else:
            cols = 2
            rows = ceil(count / 2)

        grid.styles.grid_size_columns = cols
        grid.styles.grid_size_rows = rows


def main() -> None:
    app = SystemMonitor()
    app.run()


if __name__ == "__main__":
    main()
