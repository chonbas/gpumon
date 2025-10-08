from functools import partial
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Grid, Horizontal, Vertical
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
        ("q", "quit", "Quit"),
        ("l", "toggle_log", "Toggle Log"),
        ("y", "toggle_sys", "Toggle System Stats"),
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
    #raw-log {
        height: 7;
    }
    #gpu-grid {
        grid-size: 2 2;
        grid-gutter: 1;
        height: 70%;
    }
    #bottom-row {
        height: 20%;
    }
    DataPlot, #proc-table, #raw-log {
        border: round $primary;
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
    gpu_utilization: DataPlot
    gpu_memory: DataPlot
    gpu_power: DataPlot
    process_table: DataTable
    raw_log: RichLog

    def __init__(self, gpu_id: int = 0) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.info_panel = Static(content="Querying system info...", id="info-panel")
        self.cpu_usage = DataPlot(
            name="CPU Usage (%)",
            id="cpu-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        self.gpu_memory = DataPlot(
            name=f"GPU-{self.gpu_id} Memory (%)",
            id="gpu-mem-plot",
            y_upper_lim=100,
        )
        self.gpu_power = DataPlot(
            name=f"GPU-{self.gpu_id} Power (W)",
            id="gpu-power-plot",
            value_formatter=unit_formatter(unit="W"),
        )
        self.gpu_temperature = DataPlot(
            name=f"GPU-{self.gpu_id} Temperature (°C)",
            id="gpu-temp-plot",
            value_formatter=unit_formatter(unit="°C"),
        )
        self.gpu_utilization = DataPlot(
            name=f"GPU-{self.gpu_id} Utilization (%)",
            id="gpu-util-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        self.process_table = DataTable(id="proc-table")
        self.process_table.border_title = "Processes"
        self.sys_memory = DataPlot(
            id="sys-mem-plot",
            y_upper_lim=100,
            name="System Memory (%)",
        )
        self.raw_log = RichLog(max_lines=100, id="raw-log", highlight=True, markup=True)
        self.raw_log.border_title = f"GPU-{self.gpu_id} Raw dmon Output (L to Toggle)"

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.info_panel
        with Vertical(id="plot-container"):
            with Grid(id="gpu-grid"):
                yield self.gpu_utilization
                yield self.gpu_memory
                yield self.gpu_power
                yield self.gpu_temperature
            with Horizontal(id="sys-row"):
                yield self.cpu_usage
                yield self.sys_memory
        with Horizontal(id="bottom-row"):
            yield self.raw_log
            yield self.process_table
        yield Footer()

    def on_mount(self) -> None:
        self.process_table.add_columns(*PROCESS_TABLE_COLUMNS)
        self.run_worker(
            work=partial(
                update_info_panel,
                self.gpu_id,
                self.info_panel,
                self.raw_log,
                self.gpu_memory,
                self.gpu_power,
            ),
            exclusive=True,
            group="initialization",
        )
        self.run_worker(
            work=partial(poll_cpu_percent, self.raw_log, self.cpu_usage),
            exclusive=True,
            group="cpu_polling",
        )
        self.run_worker(
            work=partial(poll_memory_percent, self.raw_log, self.sys_memory),
            exclusive=True,
            group="memory_polling",
        )
        self.run_worker(
            work=partial(
                poll_dmon_stats,
                self.gpu_id,
                self.raw_log,
                self.gpu_memory,
                self.gpu_power,
                self.gpu_temperature,
                self.gpu_utilization,
            ),
            exclusive=True,
            group="dmon_polling",
        )
        self.set_interval(
            callback=partial(update_process_list, self.raw_log, self.process_table),
            interval=DEFAULT_PROC_POLL,
        )

    def action_toggle_log(self) -> None:
        self.raw_log.display = not self.raw_log.display

    def action_toggle_sys(self) -> None:
        sys_row = self.query_one("#sys-row")
        sys_row.display = not sys_row.display
