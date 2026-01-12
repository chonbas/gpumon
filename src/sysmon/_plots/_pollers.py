import asyncio
import re
from collections.abc import Callable, Coroutine

import psutil
from textual.widgets import RichLog

from sysmon._plots import DataPlot, memory_formatter
from sysmon._utils import POLL_INTERVAL, subprocess_lifespan

DMON_BASE_CMD: list[str] = ["nvidia-smi", "dmon", "-d", str(int(POLL_INTERVAL))]


async def poll_cpu_percent(
    log: RichLog,
    cpu_plot: DataPlot,
    max_retries: int = 3,
) -> None:
    """Polls `psutil` to update CPU usage in real-time.

    This worker runs continuously and updates the CPU plot with current usage.
    Data is sent with "CPU" label to support unified utilization plots.
    Errors are logged but don't stop the polling loop.
    """

    def _cpu_poller() -> None:
        cpu_percent: float = psutil.cpu_percent(interval=None)
        cpu_plot.update_data({"CPU": cpu_percent})

    await _task_poller(
        log=log,
        name="CPU Polling",
        task=_cpu_poller,
        max_retries=max_retries,
    )


async def poll_cpu_sys_memory(
    log: RichLog,
    mem_plot: DataPlot,
    max_retries: int = 3,
) -> None:
    """Polls `psutil` to update Memory usage in real-time.

    This worker runs continuously and updates the memory plot with current usage.
    Data is sent with "System" label to support unified memory plots.
    A per-series formatter is set on the first successful poll based on total memory.
    Errors are logged but don't stop the polling loop.
    """

    def _memory_poller() -> None:
        mem = psutil.virtual_memory()
        if not mem_plot.formatter_is_set(series="System"):
            mem_plot.set_value_formatter(
                memory_formatter(total_bytes=mem.total, from_percent=True),
                series="System",
            )
        mem_plot.update_data({"System": mem.percent})

    await _task_poller(
        log=log,
        name="Memory Polling",
        task=_memory_poller,
        max_retries=max_retries,
    )


async def poll_cpu_temp(
    log: RichLog,
    temp_plot: DataPlot,
    max_retries: int = 3,
) -> None:
    """Polls `psutil` to update CPU temperature in real-time.
    Uses high and critical thresholds from the sensor for color-coded warnings.
    """

    def _temp_poller() -> None:
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

    await _task_poller(
        log=log,
        name="CPU Temp Polling",
        task=_temp_poller,
        max_retries=max_retries,
    )


async def poll_nvidia_dmon_info(
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

    async with subprocess_lifespan(
        command,
        log=log,
        name="nvidia-dmon",
    ) as proc:
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


async def _task_poller(
    log: RichLog,
    name: str,
    task: Callable[[], Coroutine[None, None, None] | None],
    max_retries: int = 3,
) -> None:
    """Helper to run a polling task with error handling and retries. If
    the task raises an exception, it is logged and retried up to max_retries.
    If task is synchronous, it is run in a thread to avoid blocking the event loop.

    Args:
        log: RichLog widget for logging output and errors
        name: Name of the polling task for logging
        task: Callable polling task to run
        max_retries: Maximum number of consecutive retries before stopping
    """
    retries = 0
    while True:
        try:
            if asyncio.iscoroutinefunction(task):
                await task()
            else:
                await asyncio.to_thread(task)
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                log.write(
                    content=f"[bold red]{name} failed too many times, "
                    "stopping.[/bold red]"
                )
                raise RuntimeError(f"{name} failed too many times") from e
            log.write(content=f"[red]{name} error:[/red] {e}")
        await asyncio.sleep(delay=POLL_INTERVAL)
