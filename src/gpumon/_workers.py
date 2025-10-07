import asyncio
import os
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psutil
from textual.widgets import DataTable, RichLog, Static

from ._defaults import DEFAULT_DMON_POLL
from ._plot import DataPlot, memory_formatter

DMON_BASE_CMD: list[str] = ["nvidia-smi", "dmon", "-d", str(DEFAULT_DMON_POLL), "-i"]
INFO_BASE_CMD: list[str] = ["nvidia-smi", "-q", "-i"]
PROCESS_QUERY_CMD: list[str] = [
    "nvidia-smi",
    "--query-compute-apps=pid,name,used_gpu_memory",
    "--format=csv,noheader,nounits",
]

PROCESS_TABLE_COLUMNS: list[str] = ["PID", "Name", "Memory", "VRAM (MiB)"]


class SysMonitorWorkerError(Exception):
    """Custom exception for SystemMonitor worker errors."""


@asynccontextmanager
async def subprocess_lifespan(
    command: list[str], /, log: RichLog, name: str = "Subprocess"
) -> AsyncGenerator[asyncio.subprocess.Process, None]:
    """A context manager to ensure a subprocess is always terminated."""
    process: asyncio.subprocess.Process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        yield process
    except Exception as e:
        log.write(content=f"[red]{name} error:[/red] {e}")
        raise SysMonitorWorkerError(f"Error in {name} - {e}") from e
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


async def poll_cpu_percent(log: RichLog, cpu_plot: DataPlot) -> None:
    """Polls `psutil` to update CPU usage in real-time."""
    while True:
        try:
            cpu_percent: float = psutil.cpu_percent(interval=None)
            cpu_plot.update_data(cpu_percent)
        except Exception as e:
            log.write(content=f"[red]CPU Polling error:[/red] {e}")
        await asyncio.sleep(delay=DEFAULT_DMON_POLL)


async def poll_memory_percent(log: RichLog, mem_plot: DataPlot) -> None:
    """Polls `psutil` to update Memory usage in real-time."""
    while True:
        try:
            mem = psutil.virtual_memory()
            if not mem_plot.formatter_is_set:
                mem_plot.set_value_formatter(memory_formatter(total_bytes=mem.total))
            mem_plot.update_data(mem.percent)
        except Exception as e:
            log.write(content=f"[red]Memory Polling error:[/red] {e}")
        await asyncio.sleep(delay=DEFAULT_DMON_POLL)


async def poll_dmon_stats(
    gpu_id: int,
    log: RichLog,
    mem_plot: DataPlot,
    power_plot: DataPlot,
    util_plot: DataPlot,
) -> None:
    """Polls `nvidia-smi dmon` to update GPU metrics in real-time."""
    command: list[str] = [*DMON_BASE_CMD, str(gpu_id)]

    async with subprocess_lifespan(command, log=log, name="nvidia-dmon") as proc:
        if proc.stdout is None or proc.stderr is None:
            log.write(content="[red]Failed to capture dmon output.[/red]")
            return

        async def consume_stdout(stdout: asyncio.StreamReader, /) -> None:
            while not stdout.at_eof():
                line: bytes = await stdout.readline()
                if not line:
                    break
                decoded: str = line.decode(encoding="utf-8", errors="ignore").strip()
                if decoded.startswith("#"):
                    continue
                log.write(content=decoded)
                parts: list[str] = re.split(pattern=r"\s+", string=decoded)
                try:
                    p, u, m = float(parts[1]), int(parts[4]), int(parts[5])
                    power_plot.update_data(p)
                    util_plot.update_data(u)
                    mem_plot.update_data(m)
                except (IndexError, ValueError):
                    pass

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
        )
        info_panel.update(content=f"{sys_info}\n{nv_info}")


async def update_process_list(log: RichLog, proc_table: DataTable) -> None:
    """Queries for running processes and updates the table."""

    async with subprocess_lifespan(
        PROCESS_QUERY_CMD, log=log, name="process-list"
    ) as proc:
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.write(
                "[red]Proc Query Failed:[/] "
                f"{stderr.decode(encoding='utf-8', errors='ignore')}"
            )
            return
        proc_table.clear()
        output: str = stdout.decode(encoding="utf-8").strip()
        if not output or "[Not Supported]" in output:
            proc_table.add_row("[i]No running processes[/i]")
        else:
            for line in output.splitlines():
                try:
                    pid, name, memory = line.split(",")
                    proc_table.add_row(
                        pid.strip(), name.strip(), f"{memory.strip()} MB"
                    )
                except ValueError:
                    log.write(f"[orange]WARN: Malformed process line:[/] {line}")


def _get_cpu_os_info() -> str:
    os_info = os.uname()
    cpu_cores: int | None = psutil.cpu_count(logical=False)
    logical_cpus: int | None = psutil.cpu_count(logical=True)
    arch: str = os_info.machine
    total_ram: float = psutil.virtual_memory().total / 1e9
    node: str = os_info.nodename
    sysname: str = os_info.sysname
    release: str = os_info.release
    return (
        f"{node}\n{sysname} | {release} | {arch}\n"
        f"CPU Cores: {cpu_cores or 'N/A'} | "
        f"Logical CPUs: {logical_cpus or 'N/A'} | "
        f"Total RAM: {total_ram: .2f} GB"
    )


def _get_gpu_info(output: str, mem_plot: DataPlot, power_plot: DataPlot) -> str:
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
        total_mem: str = mem_match.group(1).strip()
        if not mem_plot.formatter_is_set:
            mem_val, unit = total_mem.split()
            mem_val = int(mem_val.strip())
            if unit == "MiB":
                mem_val = mem_val * 1e6
            elif unit == "GiB":
                mem_val = mem_val * 1e9
            mem_plot.set_value_formatter(memory_formatter(total_bytes=mem_val))
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
