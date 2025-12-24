from __future__ import annotations

from typing import ClassVar

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Select

from sysmon._types import ProcessFilter, SortColumn, enum_from_value


class ProcessFilterInput(Widget):
    """A compound widget for filtering and sorting the process list.

    Provides a text input for fuzzy filtering and a dropdown for sort column selection.

    Attributes:
        process_filter: Current filter configuration (reactive)
    """

    DEFAULT_CSS: ClassVar[str] = """
    ProcessFilterInput {
        layout: horizontal;
        padding: 0 1;
    }
    ProcessFilterInput > Input {
        width: 1fr;
        margin-right: 1;
    }
    ProcessFilterInput > Select {
        width: 20;
        margin-right: 1;
    }
    ProcessFilterInput > Button {
        min-width: 3;
        width: 3;
        margin-right: 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding(key="escape", action="blur_filter", description="Exit filter"),
    ]

    process_filter: reactive[ProcessFilter] = reactive(
        ProcessFilter, recompose=False, init=False
    )

    class FilterChanged(Message):
        """Message emitted when the filter configuration changes.

        Attributes:
            filter: The updated ProcessFilter instance
        """

        def __init__(self, filter: ProcessFilter) -> None:
            self.filter = filter
            super().__init__()

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.process_filter = ProcessFilter()

    def compose(self) -> ComposeResult:
        """Compose the filter input components."""
        yield Select[SortColumn](
            options=[
                ("CPU %", SortColumn.CPU),
                ("Name", SortColumn.NAME),
                ("User", SortColumn.USER),
                ("PID", SortColumn.PID),
                ("Sys Mem", SortColumn.SYS_MEM),
                ("GPU Mem", SortColumn.GPU_MEM),
            ],
            id="sort-select",
            allow_blank=True,
            compact=True,
            prompt="Sort By",
        )
        yield Button("↓", id="sort-order-btn", variant="default", compact=True)
        yield Input(
            placeholder="Filter processes...",
            id="filter-input",
            compact=True,
        )

    @on(Input.Changed, "#filter-input")
    def on_filter_input_changed(self, event: Input.Changed) -> None:
        """Handle filter input text changes."""
        self.process_filter = ProcessFilter(
            query=event.value,
            sort_by=self.process_filter.sort_by,
            ascending=self.process_filter.ascending,
        )
        self.post_message(self.FilterChanged(self.process_filter))

    @on(Select.Changed, "#sort-select")
    def on_sort_select_changed(self, event: Select.Changed) -> None:
        """Handle sort column selection changes."""
        if event.value is not None and event.value != Select.BLANK:
            self.process_filter = ProcessFilter(
                query=self.process_filter.query,
                sort_by=enum_from_value(SortColumn, event.value),
                ascending=self.process_filter.ascending,
            )
            self.post_message(self.FilterChanged(self.process_filter))

    @on(Button.Pressed, "#sort-order-btn")
    def on_sort_order_pressed(self, event: Button.Pressed) -> None:
        """Toggle ascending/descending sort order."""
        new_ascending = not self.process_filter.ascending
        self.process_filter = ProcessFilter(
            query=self.process_filter.query,
            sort_by=self.process_filter.sort_by,
            ascending=new_ascending,
        )
        # Update button label to reflect new sort order
        btn = self.query_one("#sort-order-btn", Button)
        btn.label = "↑" if new_ascending else "↓"
        self.post_message(self.FilterChanged(self.process_filter))

    def action_blur_filter(self) -> None:
        """Remove focus from the filter input."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.blur()

    def focus_filter(self) -> None:
        """Focus the filter input."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.focus()

    def clear_filter(self) -> None:
        """Clear the filter input."""
        filter_input = self.query_one("#filter-input", Input)
        filter_input.value = ""
