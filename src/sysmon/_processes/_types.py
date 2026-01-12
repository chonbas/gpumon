from enum import StrEnum, auto
from typing import ClassVar, NamedTuple


class SortColumn(StrEnum):
    """Available columns for sorting process list."""

    PID = auto()
    NAME = auto()
    USER = auto()
    CPU = auto()
    SYS_MEM = auto()
    GPU_MEM = auto()


class ProcessInfo(NamedTuple):
    """Process information for display in process table."""

    pid: int
    name: str
    user: str
    cpu_percent: float
    sys_mem_mb: float
    gpu_mem: str


class ProcessFilter:
    """Filter and sort configuration for process list.

    Attributes:
        query: Fuzzy search query string
        sort_by: Column to sort by
        ascending: Sort direction
    """

    __slots__: ClassVar[tuple[str, ...]] = ("ascending", "query", "sort_by")
    query: str
    sort_by: SortColumn
    ascending: bool

    def __init__(
        self,
        query: str = "",
        sort_by: SortColumn = SortColumn.CPU,
        ascending: bool = False,
    ) -> None:
        self.query = query
        self.sort_by = sort_by
        self.ascending = ascending

    def matches(self, process: ProcessInfo) -> bool:
        """Check if process matches the filter query using fuzzy matching."""
        if not self.query:
            return True

        query_lower = self.query.lower()
        # Check PID, name, and user for fuzzy match
        searchable = f"{process.pid} {process.name} {process.user}".lower()

        # Simple fuzzy matching: check if all characters appear in order
        query_idx = 0
        for char in searchable:
            if query_idx < len(query_lower) and char == query_lower[query_idx]:
                query_idx += 1
        return query_idx == len(query_lower)

    def sort_key(self, process: ProcessInfo) -> float | str:
        """Return sort key for process based on current settings."""
        match self.sort_by:
            case SortColumn.PID:
                return process.pid
            case SortColumn.NAME:
                return process.name.lower()
            case SortColumn.USER:
                return process.user.lower()
            case SortColumn.CPU:
                return process.cpu_percent
            case SortColumn.SYS_MEM:
                return float(process.sys_mem_mb)
            case SortColumn.GPU_MEM:
                # Handle N/A values for sorting
                try:
                    return float(process.gpu_mem)
                except ValueError:
                    return -1.0 if not self.ascending else float("inf")
