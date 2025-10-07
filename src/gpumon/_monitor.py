from functools import partial
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Grid, Vertical
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
    Grid {
        grid-size: 2 3;
        grid-gutter: 1;
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
            id="cpu-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        self.gpu_memory = DataPlot(
            id="gpu-mem-plot",
            y_upper_lim=100,
        )
        self.gpu_power = DataPlot(
            id="gpu-power-plot",
            value_formatter=unit_formatter(unit="W"),
        )
        self.gpu_utilization = DataPlot(
            id="gpu-util-plot",
            y_upper_lim=100,
            value_formatter=percent_formatter(),
        )
        self.process_table = DataTable(id="proc-table")
        self.sys_memory = DataPlot(
            id="sys-mem-plot",
            y_upper_lim=100,
        )
        self.raw_log = RichLog(max_lines=100, id="raw-log", highlight=True, markup=True)

    def compose(self) -> ComposeResult:
        self.cpu_usage.border_title = "CPU Usage (%)"
        self.gpu_memory.border_title = f"GPU-{self.gpu_id} Memory Util (%)"
        self.gpu_power.border_title = f"GPU-{self.gpu_id} Power (W)"
        self.gpu_utilization.border_title = f"GPU-{self.gpu_id} Utilization (%)"
        self.process_table.border_title = "Processes"
        self.raw_log.border_title = f"GPU-{self.gpu_id} Raw dmon Output (L to Toggle)"
        self.sys_memory.border_title = "System Memory Util (%)"
        yield Header()
        yield self.info_panel
        with Vertical(id="plot-container"), Grid(id="plots-grid"):
            yield self.cpu_usage
            yield self.sys_memory
            yield self.gpu_utilization
            yield self.gpu_memory
            yield self.gpu_power
            yield self.process_table
        yield self.raw_log
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
