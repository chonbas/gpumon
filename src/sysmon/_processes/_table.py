from __future__ import annotations

from typing import ClassVar

import psutil
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import DataTable, RichLog

from ._filter import ProcessFilterInput
from ._types import ProcessFilter, ProcessInfo
from ._utils import get_processes, sort_and_filter_processes

INTERACTION_TIMEOUT: float = 5.0
PROCESS_TABLE_COLUMNS: list[str] = [
    "PID",
    "Name",
    "User",
    "CPU %",
    "Sys Mem",
    "GPU Mem",
]


class ProcessTable(Widget):
    DEFAULT_CSS: ClassVar[str] = """
    #proc-container {
        width: 1fr;
    }
    #proc-table {
        background: $background;
        border: round $primary;
        height: 1fr;
    }
    #proc-table.user-interacting {
        border: round $warning;
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

    _filter: ProcessFilter
    """Active filter being applied to the process list."""
    _filter_input: ProcessFilterInput
    """The filter input widget."""
    _table: DataTable
    """Actual data table widget."""
    _user_interacting: bool
    """Whether the user is currently interacting with the table."""
    _interaction_timer: Timer | None
    """Timer to reset interaction state after timeout."""

    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        markup: bool = True,
    ) -> None:
        super().__init__(
            *children,
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
            markup=markup,
        )
        self._filter = ProcessFilter()
        self._filter_input = ProcessFilterInput()
        self._table = DataTable(id="proc-table", cursor_type="row")
        self._table.border_title = "Processes"
        self._user_interacting = False
        self._interaction_timer = None

    def on_mount(self) -> None:
        self._table.add_columns(*PROCESS_TABLE_COLUMNS)

    def compose(self) -> ComposeResult:
        with Vertical(id="proc-container"):
            yield self._filter_input
            yield self._table

    def focus_filter(self) -> None:
        """Focus the filter input."""
        self._filter_input.focus_filter()

    @property
    def is_user_interacting(self) -> bool:
        """Whether the user is currently interacting with the table.

        When True, automatic updates to the process list are paused
        to allow the user to inspect entries without the table
        being refreshed and reordered.
        """
        return self._user_interacting

    @is_user_interacting.setter
    def is_user_interacting(self, interacting: bool) -> None:
        """Set the interaction state and update visual feedback.

        Args:
            interacting: Whether the user is interacting with the table.
        """
        if interacting == self._user_interacting:
            return

        self._user_interacting = interacting
        if interacting:
            self._table.add_class("user-interacting")
            self._table.border_title = "Processes [dim](paused)[/dim]"
        else:
            self._table.remove_class("user-interacting")
            self._table.border_title = "Processes"

    def kill_process(self, log: RichLog) -> None:
        """Kills the selected process."""
        try:
            row_key = self._table.coordinate_to_cell_key(
                self._table.cursor_coordinate
            ).row_key
            if row_key and row_key.value is not None:
                pid = int(row_key.value)
                psutil.Process(pid).kill()
                log.write(f"[green]Killed process {pid}[/green]")
        except Exception as e:
            log.write(f"[red]Failed to kill process: {e}[/red]")

    def filter_process_list(self) -> None:
        """Re-apply the current filter and sort to the existing table data.

        This method re-sorts and filters the current table contents without
        querying for new process data. Used when the user changes filter/sort
        settings while the table is paused.
        """
        # If table is empty or has placeholder text, nothing to do
        if self._table.row_count == 0:
            return

        # Check if we have a placeholder row (no actual data)
        try:
            first_row = self._table.get_row_at(0)
            if (
                first_row
                and len(first_row) == 1
                and str(first_row[0]).startswith("[i]")
            ):
                return
        except Exception:
            return
        # Sort by configured column
        sorted_procs: list[ProcessInfo] = sort_and_filter_processes(
            self._table, filter=self._filter
        )
        self._update_table_content(sorted_procs)

    async def update_process_list(self, log: RichLog) -> None:
        """Queries for running processes and updates the table.

        This function is called periodically to refresh the process list.
        It queries nvidia-smi for GPU processes and psutil for all processes,
        merging the data and displaying the top processes.

        If the user is currently interacting with the table (navigating rows,
        selecting entries), the update is skipped to prevent disrupting
        their inspection.

        Args:
            log: RichLog widget for logging errors
        """
        # Skip update if user is interacting with the table
        if self._user_interacting:
            return
        sorted_procs = await get_processes(log=log, filter=self._filter)
        self._update_table_content(sorted_procs)

    def _reset_interaction_timer(self) -> None:
        """Reset the interaction timeout timer.

        This is called whenever the user interacts with the table.
        After INTERACTION_TIMEOUT seconds of inactivity, the table
        will resume automatic updates.
        """
        # Cancel any existing timer
        if self._interaction_timer is not None:
            self._interaction_timer.stop()
            self._interaction_timer = None

        # Mark as interacting
        self.is_user_interacting = True
        # Start a new timer to reset after timeout
        self._interaction_timer = self.set_timer(
            delay=INTERACTION_TIMEOUT,
            callback=self._on_interaction_timeout,
            name="interaction-timeout",
        )

    def _on_interaction_timeout(self) -> None:
        """Called when the interaction timeout expires."""
        self._interaction_timer = None
        self.is_user_interacting = False

    @on(DataTable.RowHighlighted, "#proc-table")
    def _on_row_highlighted(self) -> None:
        """Handle row highlight events (cursor movement)."""
        if self._interaction_timer is not None:
            self._reset_interaction_timer()

    @on(DataTable.RowLabelSelected, "#proc-table")
    def _on_row_label_selected(self) -> None:
        """Handle row highlight events (cursor movement)."""
        self._reset_interaction_timer()

    @on(DataTable.RowSelected, "#proc-table")
    def _on_row_selected(self) -> None:
        """Handle row selection events (enter key)."""
        self._reset_interaction_timer()

    @on(ProcessFilterInput.FilterChanged)
    def _on_filter_changed(self, event: ProcessFilterInput.FilterChanged) -> None:
        """Handle filter changes from the ProcessFilterInput widget.

        Updates the internal filter and re-applies it to the current data.
        This allows re-ordering even when paused, since the user explicitly
        requested a filter/sort change.
        """
        self._filter = event.filter
        self.filter_process_list()

    def _update_table_content(self, processes: list[ProcessInfo]) -> None:
        """Update the table content with the given list of processes."""
        self._table.clear()
        if not processes:
            if self._filter.query:
                self._table.add_row("[i]No matching processes[/i]")
            else:
                self._table.add_row("[i]No running processes[/i]")
        else:
            for p in processes:
                self._table.add_row(
                    str(p.pid),
                    p.name,
                    p.user,
                    f"{p.cpu_percent:.1f}",
                    f"{p.sys_mem_mb:.1f} MB",
                    f"{p.gpu_mem} MB" if p.gpu_mem != "N/A" else "-",
                    key=str(p.pid),
                )
