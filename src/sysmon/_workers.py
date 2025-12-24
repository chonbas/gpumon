import asyncio
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psutil
from textual.widgets import DataTable, RichLog, Static

from sysmon._defaults import DEFAULT_DMON_POLL, GIB, MIB
from sysmon._plot import DataPlot, memory_formatter
from sysmon._types import ProcessFilter, ProcessInfo

DMON_BASE_CMD: list[str] = ["nvidia-smi", "dmon", "-d", str(DEFAULT_DMON_POLL)]
INFO_BASE_CMD: list[str] = ["nvidia-smi", "-q", "-i"]
PROCESS_QUERY_CMD: list[str] = [
    "nvidia-smi",
    "--query-compute-apps=pid,used_gpu_memory",
    "--format=csv,noheader,nounits",
]

PROCESS_TABLE_COLUMNS: list[str] = [
    "PID",
    "Name",
    "User",
    "CPU %",
    "Sys Mem",
    "GPU Mem",
]


@asynccontextmanager
async def subprocess_lifespan(
    command: list[str],
    /,
    log: RichLog,
    name: str = "Subprocess",
) -> AsyncGenerator[asyncio.subprocess.Process, None]:
    """A context manager to ensure a subprocess is always terminated.
    Args:
        command: The command to execute as a list of strings
        log: RichLog widget for logging errors
        name: Name of the subprocess for logging
    """
    process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield process
    except Exception as e:
        log.write(content=f"[red]{name} error:[/red] {e}")
        raise RuntimeError(f"Error in {name} - {e}") from e
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def subprocess_communicate(
    command: list[str],
    /,
    log: RichLog,
    name: str = "Subprocess",
    timeout: float = 5.0,
) -> str | None:
    async with subprocess_lifespan(command, log=log, name=name) as proc:
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            log.write("[orange]Process query timed out[/orange]")
            return
    if proc.returncode != 0:
        log.write(
            f"[red]{name} failed:[/] {stderr.decode(encoding='utf-8', errors='ignore')}"
        )
        return
    return stdout.decode(encoding="utf-8").strip()


async def poll_cpu_percent(
    log: RichLog, cpu_plot: DataPlot, max_retries: int = 3
) -> None:
    """Polls `psutil` to update CPU usage in real-time.

    This worker runs continuously and updates the CPU plot with current usage.
    Data is sent with "CPU" label to support unified utilization plots.
    Errors are logged but don't stop the polling loop.
    """
    retries = 0
    while True:
        try:
            cpu_percent: float = psutil.cpu_percent(interval=None)
            cpu_plot.update_data({"CPU": cpu_percent})
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                log.write(
                    content="[bold red]CPU polling failed too many times, "
                    "stopping.[/bold red]"
                )
                raise RuntimeError("CPU polling failed too many times") from e
            log.write(content=f"[red]CPU Polling error:[/red] {e}")
        await asyncio.sleep(delay=DEFAULT_DMON_POLL)


async def poll_cpu_temp(
    log: RichLog, temp_plot: DataPlot, max_retries: int = 3
) -> None:
    """Polls `psutil` to update CPU temperature in real-time.

    Uses high and critical thresholds from the sensor for color-coded warnings.
    """
    retries = 0
    while True:
        try:
            temps = psutil.sensors_temperatures()
            data: dict[str, float] = {}
            thresholds: dict[str, tuple[float | None, float | None]] = {}

            if "coretemp" in temps:
                for entry in temps["coretemp"]:
                    if "Package" in entry.label:
                        data["CPU"] = entry.current
                        # Capture high and critical thresholds for color coding
                        thresholds["CPU"] = (entry.high, entry.critical)
                        break
            if data:
                temp_plot.update_data(data, thresholds=thresholds)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                log.write(
                    content="[bold red]CPU Temp polling failed too many times, "
                    "stopping.[/bold red]"
                )
                raise RuntimeError("CPU Temp polling failed too many times") from e
            log.write(content=f"[red]CPU Temp Polling error:[/red] {e}")
        await asyncio.sleep(delay=DEFAULT_DMON_POLL)


async def poll_system_memory(
    log: RichLog, mem_plot: DataPlot, max_retries: int = 3
) -> None:
    """Polls `psutil` to update Memory usage in real-time.

    This worker runs continuously and updates the memory plot with current usage.
    Data is sent with "System" label to support unified memory plots.
    A per-series formatter is set on the first successful poll based on total memory.
    Errors are logged but don't stop the polling loop.
    """
    retries = 0

    while True:
        try:
            mem = psutil.virtual_memory()
            if not mem_plot.formatter_is_set(series="System"):
                mem_plot.set_value_formatter(
                    memory_formatter(total_bytes=mem.total, from_percent=True),
                    series="System",
                )
            mem_plot.update_data({"System": mem.percent})
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                log.write(
                    content="[bold red]Memory polling failed too many times, "
                    "stopping.[/bold red]"
                )
                raise
            log.write(content=f"[red]Memory Polling error:[/red] {e}")
        await asyncio.sleep(delay=DEFAULT_DMON_POLL)


async def poll_dmon_stats(
    log: RichLog,
    mem_plot: DataPlot,
    power_plot: DataPlot,
    temp_plot: DataPlot,
    util_plot: DataPlot,
) -> None:
    """Polls `nvidia-smi dmon` to update GPU metrics in real-time.
    This worker streams output from nvidia-smi dmon command and parses
    GPU metrics (power, temperature, utilization, memory) to update plots.
    The subprocess runs continuously until the worker is cancelled.
    Stdout and stderr are consumed concurrently to prevent blocking.
    Args:
        log: RichLog widget for logging output and errors
        mem_plot: Plot widget for GPU memory usage
        power_plot: Plot widget for GPU power consumption
        temp_plot: Plot widget for GPU temperature
        util_plot: Plot widget for GPU utilization
    """
    command: list[str] = [*DMON_BASE_CMD]

    async with subprocess_lifespan(command, log=log, name="nvidia-dmon") as proc:
        if proc.stdout is None or proc.stderr is None:
            log.write(content="[red]Failed to capture dmon output.[/red]")
            return

        async def consume_stdout(stdout: asyncio.StreamReader, /) -> None:
            parse_error_count = 0
            max_parse_errors = 20

            while not stdout.at_eof():
                line: bytes = await stdout.readline()
                if not line:
                    break
                decoded: str = line.decode(encoding="utf-8", errors="ignore").strip()
                if decoded.startswith("#"):
                    continue
                parts: list[str] = re.split(pattern=r"\s+", string=decoded)
                try:
                    idx = parts[0]
                    pwr, temp, util, mem = (
                        float(parts[1]),
                        int(parts[2]),
                        int(parts[4]),
                        int(parts[5]),
                    )
                    gpu_label = f"GPU-{idx}"
                    power_plot.update_data({gpu_label: pwr})
                    util_plot.update_data({gpu_label: util})
                    temp_plot.update_data({gpu_label: temp})
                    mem_plot.update_data({gpu_label: mem})
                    parse_error_count = 0  # Reset on success
                except (IndexError, ValueError) as e:
                    parse_error_count += 1
                    if parse_error_count >= max_parse_errors:
                        log.write(
                            content=f"[bold red]Too many parse errors in dmon "
                            f"output: {e}[/bold red]"
                        )

        async def consume_stderr(stderr: asyncio.StreamReader, /) -> None:
            while not stderr.at_eof():
                line: bytes = await stderr.readline()
                if not line:
                    break
                log.write(
                    content="[bold red]DMON STDERR:[/bold red] "
                    f"{line.decode(encoding='utf-8', errors='ignore').strip()}"
                )

        await asyncio.gather(consume_stdout(proc.stdout), consume_stderr(proc.stderr))


async def update_info_panel(
    gpu_id: int,
    info_panel: Static,
    log: RichLog,
    mem_plot: DataPlot,
    power_plot: DataPlot,
) -> None:
    """Query nvidia-smi for static GPU info to set info panel."""
    command: list[str] = [*INFO_BASE_CMD, str(gpu_id)]
    sys_info: str = _get_cpu_os_info()
    info_panel.update(content=f"{sys_info}\nQuerying GPU info...")

    async with subprocess_lifespan(command, log=log, name="info-panel") as proc:
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(encoding="utf-8", errors="ignore"))

        output: str = stdout.decode()
        nv_info: str = _get_gpu_info(
            output=output,
            mem_plot=mem_plot,
            power_plot=power_plot,
            gpu_id=gpu_id,
        )
        info_panel.update(content=f"{sys_info}\n{nv_info}")


async def update_process_list(
    log: RichLog,
    proc_table: DataTable,
    process_filter: ProcessFilter | None = None,
) -> None:
    """Queries for running processes and updates the table.

    This function is called periodically to refresh the process list.
    It queries nvidia-smi for GPU processes and psutil for all processes,
    merging the data and displaying the top processes.

    Args:
        log: RichLog widget for logging errors
        proc_table: DataTable widget to display process information
        process_filter: Optional filter/sort configuration
    """
    if process_filter is None:
        process_filter = ProcessFilter()

    # 1. Get GPU processes
    gpu_procs: dict[int, str] = {}
    try:
        output: str | None = await subprocess_communicate(
            PROCESS_QUERY_CMD, log=log, name="process-list"
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
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception as e:
        log.write(f"[red]Process list update failed:[/red] {e}")
        return

    # 3. Apply filter
    filtered_procs = [p for p in procs if process_filter.matches(p)]

    # 4. Sort by configured column
    sorted_procs = sorted(
        filtered_procs,
        key=process_filter.sort_key,
        reverse=not process_filter.ascending,
    )

    # Take top 50
    top_procs = sorted_procs[:50]

    # Update Table
    proc_table.clear()
    if not top_procs:
        if process_filter.query:
            proc_table.add_row("[i]No matching processes[/i]")
        else:
            proc_table.add_row("[i]No running processes[/i]")
    else:
        for p in top_procs:
            proc_table.add_row(
                str(p.pid),
                p.name,
                p.user,
                f"{p.cpu_percent:.1f}",
                f"{p.sys_mem_mb:.1f} MB",
                f"{p.gpu_mem} MB" if p.gpu_mem != "N/A" else "-",
                key=str(p.pid),
            )


def _get_cpu_os_info() -> str:
    os_info = os.uname()
    cpu_cores: int | None = psutil.cpu_count(logical=False)
    logical_cpus: int | None = psutil.cpu_count(logical=True)
    arch: str = os_info.machine
    total_ram: float = psutil.virtual_memory().total / GIB
    node: str = os_info.nodename
    sysname: str = os_info.sysname
    release: str = os_info.release
    return (
        f"{node}\n{sysname} | {release} | {arch}\n"
        f"CPU Cores: {cpu_cores or 'N/A'} | "
        f"Logical CPUs: {logical_cpus or 'N/A'} | "
        f"Total RAM: {total_ram: .2f} GiB"
    )


def _get_gpu_info(
    output: str, mem_plot: DataPlot, power_plot: DataPlot, gpu_id: int = 0
) -> str:
    """Parse GPU info from nvidia-smi output and set formatters.

    Args:
        output: Raw nvidia-smi -q output
        mem_plot: Memory plot to set formatter for
        power_plot: Power plot to set y-axis limit
        gpu_id: GPU index for setting per-series formatter
    """
    name, driver, mem, cuda = "N/A", "N/A", "N/A", "N/A"
    name_match: re.Match[str] | None = re.search(
        pattern=r"Product Name\s+:\s+(.*)", string=output
    )
    driver_match: re.Match[str] | None = re.search(
        pattern=r"Driver Version\s+:\s+([\d.]+)", string=output
    )
    cuda_match: re.Match[str] | None = re.search(
        pattern=r"CUDA Version\s+:\s+([\d.]+)", string=output
    )
    mem_match: re.Match[str] | None = re.search(
        pattern=r"FB Memory Usage[\s\S]*?Total\s+:\s+([\d]+\s+MiB)",
        string=output,
    )
    max_watt_match: re.Match[str] | None = re.search(
        pattern=r"Max Power Limit\s+:\s+([\d.]+\s+W)",
        string=output,
    )
    if name_match:
        name: str = name_match.group(1).strip()
    if driver_match:
        driver: str = driver_match.group(1).strip()
    if cuda_match:
        cuda: str = cuda_match.group(1).strip()
    if mem_match:
        mem: str = mem_match.group(1).strip()
        gpu_series = f"GPU-{gpu_id}"
        if not mem_plot.formatter_is_set(series=gpu_series):
            mem_val, unit = mem.split()
            tot_mem: float = float(mem_val.strip())
            if unit == "MiB":
                tot_mem = tot_mem * MIB
            elif unit == "GiB":
                tot_mem = tot_mem * GIB
            mem_plot.set_value_formatter(
                memory_formatter(total_bytes=tot_mem, from_percent=True),
                series=gpu_series,
            )
    if max_watt_match:
        watt_str: str = max_watt_match.group(1).strip()
        watt_val, _ = watt_str.split()
        try:
            watt_val_f: float = float(watt_val)
            power_plot.set_y_upper_lim(watt_val_f * 1.1)
        except ValueError:
            pass
    info_text: str = f"{name} | Driver: {driver} | CUDA: {cuda} | VRAM: {mem}"
    return info_text
