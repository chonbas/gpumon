import psutil
from textual.widgets import DataTable, RichLog

from sysmon._utils import MIB, subprocess_communicate

from ._filter import ProcessFilter
from ._types import ProcessInfo

PROCESS_QUERY_CMD: list[str] = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]


async def get_processes(
    log: RichLog, filter: ProcessFilter | None = None
) -> list[ProcessInfo]:
    """Fetch the list of current system processes, including GPU memory usage,
    and apply optional filtering and sorting.
    Args:
        log: RichLog widget for logging errors
        filter: Optional ProcessFilter to apply filtering and sorting
    Returns:
        A list of ProcessInfo instances representing current processes
    """

    # 1. Get GPU processes
    gpu_procs: dict[int, str] = {}
    try:
        output: str | None = await subprocess_communicate(
            PROCESS_QUERY_CMD,
            log=log,
            name="process-list",
        )
        if output and "[Not Supported]" not in output:
            for line in output.splitlines():
                try:
                    parts = line.split(",")
                    if len(parts) >= 2:
                        pid = int(parts[0].strip())
                        mem = parts[1].strip()  # MB
                        gpu_procs[pid] = mem
                except ValueError:
                    continue
    except Exception as e:
        log.write(f"[orange]WARN: GPU process query failed: {e}[/orange]")

    # 2. Get all processes via psutil
    procs: list[ProcessInfo] = []
    try:
        for p in psutil.process_iter(
            ["pid", "name", "username", "memory_info", "cpu_percent"]
        ):
            try:
                p_info = p.info
                pid = p_info["pid"]
                gpu_mem = gpu_procs.get(pid, "N/A")

                mem_info = p_info["memory_info"]
                sys_mem = mem_info.rss / MIB if mem_info else 0.0
                proc = ProcessInfo(
                    pid=pid,
                    name=p_info["name"] or "",
                    user=p_info["username"] or "",
                    cpu_percent=p_info["cpu_percent"] or 0.0,
                    sys_mem_mb=sys_mem,
                    gpu_mem=gpu_mem,
                )
                procs.append(proc)
            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
                psutil.ZombieProcess,
            ):
                pass
    except Exception as e:
        log.write(f"[red]Process list update failed:[/red] {e}")
        return []

    if not procs or filter is None:
        return procs
    return _sort_filter(procs, filter)


def sort_and_filter_processes(
    table: DataTable,
    filter: ProcessFilter | None = None,
) -> list[ProcessInfo]:
    """Sort and filter processes currently displayed in the DataTable.
    Args:
        table: DataTable widget containing process data
        filter: Optional ProcessFilter to apply filtering and sorting
    Returns:
        A list of ProcessInfo instances after applying filtering and sorting
    """
    # Collect current process data from the table
    current_procs: list[ProcessInfo] = []
    for row_key in list(table.rows.keys()):
        try:
            row_data = table.get_row(row_key)
            if row_data and len(row_data) >= 6:
                pid = int(row_data[0])
                name = str(row_data[1])
                user = str(row_data[2])
                cpu_percent = float(str(row_data[3]))
                # Parse sys_mem, removing " MB" suffix
                sys_mem_str = str(row_data[4]).replace(" MB", "")
                sys_mem = float(sys_mem_str) if sys_mem_str else 0.0
                # Parse gpu_mem, handling "-" and " MB" suffix
                gpu_mem_str = str(row_data[5])
                if gpu_mem_str == "-":
                    gpu_mem = "N/A"
                else:
                    gpu_mem = gpu_mem_str.replace(" MB", "")

                proc = ProcessInfo(
                    pid=pid,
                    name=name,
                    user=user,
                    cpu_percent=cpu_percent,
                    sys_mem_mb=sys_mem,
                    gpu_mem=gpu_mem,
                )
                current_procs.append(proc)
        except (ValueError, IndexError):
            continue

    if not current_procs or filter is None:
        return current_procs
    return _sort_filter(current_procs, filter)


def _sort_filter(procs: list[ProcessInfo], filter: ProcessFilter) -> list[ProcessInfo]:
    """Sort and filter a list of ProcessInfo based on the given ProcessFilter.

    Args:
        procs: List of ProcessInfo instances to sort and filter
        filter: ProcessFilter instance defining filtering and sorting criteria
    Returns:
        A new list of ProcessInfo instances after applying filtering and sorting
    """
    filtered_procs = (
        [p for p in procs if filter.matches(p)] if filter is not None else procs
    )
    return sorted(filtered_procs, key=filter.sort_key, reverse=not filter.ascending)
