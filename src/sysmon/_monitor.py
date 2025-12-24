from functools import partial
from math import ceil
from typing import ClassVar

import psutil
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Grid, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Static

from sysmon._defaults import DEFAULT_PROC_POLL
from sysmon._plot import DataPlot, percent_formatter, unit_formatter
from sysmon._types import ProcessFilter
from sysmon._widgets import ProcessFilterInput
from sysmon._workers import (
    PROCESS_TABLE_COLUMNS,
    poll_cpu_percent,
    poll_cpu_temp,
    poll_dmon_stats,
    poll_system_memory,
    update_info_panel,
    update_process_list,
)


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
        text-align: center;
        text-style: bold;
    }

    #raw-log {
        height: 10%;
    }
    #proc-container {
        height: 25%;
        width: 1fr;
    }
    DataPlot, #proc-table, #raw-log {
        border: round $primary;
    }
    DataPlot:disabled {
        opacity: 0.0;
    }
    #proc-table {
        background: $background;
        height: 1fr;
    }
    DataTable > .datatable--header { background: $surface-darken-2; }
    DataTable > .datatable--odd-row { background: $surface; }
    DataTable > .datatable--even-row { background: $surface-darken-1; }
    ProcessFilterInput {
        dock: top;
        height: 3;
        border-bottom: solid $primary;
    }
    """

    gpu_id: int
    info_panel: Static
    process_table: DataTable
    raw_log: RichLog
    plots: list[DataPlot]
    process_filter: ProcessFilter
    filter_widget: ProcessFilterInput

    def __init__(self, gpu_id: int = 0) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.info_panel = Static(content="Querying system info...", id="info-panel")
        self.process_filter = ProcessFilter()

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

        self.process_table = DataTable(id="proc-table", cursor_type="row")
        self.process_table.border_title = "Processes"
        self.filter_widget = ProcessFilterInput(id="process-filter")
        self.raw_log = RichLog(max_lines=10, id="raw-log", highlight=True, markup=True)
        self.raw_log.border_title = "Log (L to Toggle)"
        self.raw_log.display = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield self.info_panel
        with Grid(id="gpu-grid"):
            for plot in self.plots:
                yield plot
        with Vertical(id="proc-container"):
            yield self.filter_widget
            yield self.process_table
        yield self.raw_log
        yield Footer()

    def on_mount(self) -> None:
        self.theme = "gruvbox"
        self.process_table.add_columns(*PROCESS_TABLE_COLUMNS)
        self._update_plot_grid_layout()

        unified_memory, unified_utilization, unified_temperature, gpu_power = self.plots

        # Initialize workers for collapsed unified plots
        self.run_worker(
            work=partial(
                update_info_panel,
                self.gpu_id,
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
            work=partial(poll_system_memory, self.raw_log, unified_memory),
            exclusive=True,
            group="memory_polling",
        )

        # GPU stats feed into unified plots
        self.run_worker(
            work=partial(
                poll_dmon_stats,
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
            interval=DEFAULT_PROC_POLL,
        )

    async def _refresh_process_list(self) -> None:
        """Wrapper to refresh process list with current filter state."""
        await update_process_list(
            self.raw_log,
            self.process_table,
            self.process_filter,
        )

    @on(ProcessFilterInput.FilterChanged)
    def on_filter_changed(self, event: ProcessFilterInput.FilterChanged) -> None:
        """Handle process filter changes and immediately refresh the list."""
        self.process_filter = event.filter
        # Trigger immediate refresh with new filter
        self.call_later(self._refresh_process_list)

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

    def action_toggle_log(self) -> None:
        self.raw_log.display = not self.raw_log.display

    def action_toggle_plot(self, plot_index: int) -> None:
        """Toggles the display of a plot and updates the grid layout."""
        if 0 <= plot_index < len(self.plots):
            plot = self.plots[plot_index]
            plot.display = not plot.display
            self._update_plot_grid_layout()

    def action_focus_filter(self) -> None:
        """Focus the process filter input."""
        self.filter_widget.focus_filter()

    def action_kill_process(self) -> None:
        """Kills the selected process."""
        try:
            row_key = self.process_table.coordinate_to_cell_key(
                self.process_table.cursor_coordinate
            ).row_key
            if row_key:
                pid = int(row_key.value)
                psutil.Process(pid).kill()
                self.raw_log.write(f"[green]Killed process {pid}[/green]")
        except Exception as e:
            self.raw_log.write(f"[red]Failed to kill process: {e}[/red]")
