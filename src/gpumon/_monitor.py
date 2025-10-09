from functools import partial
from math import ceil
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Horizontal
from textual.reactive import Reactive, var
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from ._defaults import DEFAULT_PROC_POLL
from ._plot import DataPlot, percent_formatter, unit_formatter
from ._workers import (
    PROCESS_TABLE_COLUMNS,
    poll_cpu_percent,
    poll_dmon_stats,
    poll_memory_percent,
    update_info_panel,
    update_process_list,
)


class SystemMonitor(App):
    """Monitors NVIDIA GPU statistics, as well as CPU and Memory usage
    in real-time using `nvidia-smi` and `psutil`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="l", action="toggle_log", description="Toggle Log"),
        Binding(key="1", action="toggle_plot(0)", description="GPU Util", show=True),
        Binding(key="2", action="toggle_plot(1)", description="GPU Mem", show=True),
        Binding(key="3", action="toggle_plot(2)", description="GPU Power", show=True),
        Binding(key="4", action="toggle_plot(3)", description="GPU Temp", show=True),
        Binding(key="5", action="toggle_plot(4)", description="CPU Usage", show=True),
        Binding(key="6", action="toggle_plot(5)", description="Sys Mem", show=True),
    ]
    TITLE = "SystemMonitor"

    CSS = """
    #info-panel {
        height: auto;
        padding: 0 1;
        color: $primary;
        border: round $primary;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }

    #bottom-row {
        height: 20%;
    }
    DataPlot, #proc-table, #raw-log {
        border: round $primary;
    }
    DataPlot:disabled {
        opacity: 0.0;
    }
    #proc-table {
        background: $background;
        height: 100%;
    }
    DataTable > .datatable--header { background: $surface-darken-2; }
    DataTable > .datatable--odd-row { background: $surface; }
    DataTable > .datatable--even-row { background: $surface-darken-1; }
    """

    gpu_id: int
    info_panel: Static
    process_table: DataTable
    raw_log: RichLog
    plots: list[DataPlot]

    def __init__(self, gpu_id: int = 0) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.info_panel = Static(content="Querying system info...", id="info-panel")

        gpu_utilization = DataPlot(
            name=f"GPU-{self.gpu_id} Utilization (%)",
            id="gpu-util-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        gpu_memory = DataPlot(
            name=f"GPU-{self.gpu_id} Memory (%)",
            id="gpu-mem-plot",
            y_upper_lim=100,
        )
        gpu_power = DataPlot(
            name=f"GPU-{self.gpu_id} Power (W)",
            id="gpu-power-plot",
            value_formatter=unit_formatter(unit="W"),
        )
        gpu_temperature = DataPlot(
            name=f"GPU-{self.gpu_id} Temperature (°C)",
            id="gpu-temp-plot",
            value_formatter=unit_formatter(unit="°C"),
        )
        cpu_usage = DataPlot(
            name="CPU Usage (%)",
            id="cpu-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        sys_memory = DataPlot(
            id="sys-mem-plot",
            y_upper_lim=100,
            name="System Memory (%)",
        )

        self.plots = [
            gpu_utilization,
            gpu_memory,
            gpu_power,
            gpu_temperature,
            cpu_usage,
            sys_memory,
        ]

        self.process_table = DataTable(id="proc-table")
        self.process_table.border_title = "Processes"
        self.raw_log = RichLog(max_lines=100, id="raw-log", highlight=True, markup=True)
        self.raw_log.border_title = f"GPU-{self.gpu_id} Raw dmon Output (L to Toggle)"

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.info_panel
        with Grid(id="gpu-grid"):
            for plot in self.plots:
                yield plot
        with Horizontal(id="bottom-row"):
            yield self.raw_log
            yield self.process_table
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "gruvbox"
        self.process_table.add_columns(*PROCESS_TABLE_COLUMNS)
        self._update_plot_grid_layout()

        (
            gpu_utilization,
            gpu_memory,
            gpu_power,
            gpu_temperature,
            cpu_usage,
            sys_memory,
        ) = self.plots

        self.run_worker(
            work=partial(
                update_info_panel,
                self.gpu_id,
                self.info_panel,
                self.raw_log,
                gpu_memory,
                gpu_power,
            ),
            exclusive=True,
            group="initialization",
        )
        self.run_worker(
            work=partial(poll_cpu_percent, self.raw_log, cpu_usage),
            exclusive=True,
            group="cpu_polling",
        )
        self.run_worker(
            work=partial(poll_memory_percent, self.raw_log, sys_memory),
            exclusive=True,
            group="memory_polling",
        )
        self.run_worker(
            work=partial(
                poll_dmon_stats,
                self.gpu_id,
                self.raw_log,
                gpu_memory,
                gpu_power,
                gpu_temperature,
                gpu_utilization,
            ),
            exclusive=True,
            group="dmon_polling",
        )
        self.set_interval(
            callback=partial(update_process_list, self.raw_log, self.process_table),
            interval=DEFAULT_PROC_POLL,
        )

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
        elif count == 3:
            cols = 3
            rows = 1
        elif count == 4:
            cols = 2
            rows = 2
        else:
            cols = 3
            rows = ceil(count / 3)

        grid.styles.grid_size_columns = cols
        grid.styles.grid_size_rows = rows

    def action_toggle_log(self) -> None:
        self.raw_log.display = not self.raw_log.display

    def action_toggle_plot(self, plot_index: int) -> None:
        """Toggles the display of a plot and updates the grid layout."""
        if 0 <= plot_index < len(self.plots):
            plot = self.plots[plot_index]
            plot.display = not plot.display
            self._update_plot_grid_layout()
