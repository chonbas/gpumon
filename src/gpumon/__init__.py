import argparse
import asyncio
import os
import re
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import ClassVar

import pytz
from textual.app import App, ComposeResult
from textual.binding import BindingType
from textual.containers import Grid, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Static
from textual_plotext import PlotextPlot

DEFAULT_HISTORY = 1000
DEFAULT_DMON_POLL = 1
DEFAULT_PROC_POLL = 3
PROCESS_QUERY_CMD: list[str] = [
    "nvidia-smi",
    "--query-compute-apps=pid,name,used_gpu_memory",
    "--format=csv,noheader,nounits",
]
DMON_BASE_CMD: list[str] = ["nvidia-smi", "dmon", "-d", str(DEFAULT_DMON_POLL), "-i"]
INFO_BASE_CMD: list[str] = ["nvidia-smi", "-q", "-i"]

LOCAL_TIMEZONE = os.getenv("LOCAL_TIMEZONE", "US/Arizona")


@asynccontextmanager
async def subprocess_lifespan(
    command: list[str], /, log: RichLog
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
        log.write(f"[red]Subprocess error:[/red] {e}")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class GPUPlot(PlotextPlot):
    """A reactive plotext widget for displaying a single GPU metric."""

    marker: str = "braille"
    data: deque[tuple[datetime, float]]
    history_size: int
    dmon_interval: int
    proc_interval: int
    tz: pytz.BaseTzInfo

    def __init__(
        self,
        tz: str = LOCAL_TIMEZONE,
        history_size: int = DEFAULT_HISTORY,
        dmon_interval: int = DEFAULT_DMON_POLL,
        proc_interval: int = DEFAULT_PROC_POLL,
    ) -> None:
        super().__init__()
        self.data = deque(maxlen=DEFAULT_HISTORY)
        self.history_size = history_size
        self.dmon_interval = dmon_interval
        self.proc_interval = proc_interval
        self.tz = pytz.timezone(tz)

    def update_data(self, value: float, /) -> None:
        """Appends the new data point with the current timestamp."""
        self.data.append((datetime.now(self.tz), value))
        self.draw_plot()
        self.refresh()

    def draw_plot(self) -> None:
        """Draws the plot with a time-based x-axis."""
        self.plt.clf()
        self.plt.theme(theme="dark")
        if self.data:
            x_times, y_values = zip(*self.data, strict=False)
            self.plt.plot(y_values, marker=self.marker, color="cyan")
            self.plt.xticks(
                list(range(len(x_times))), labels=[f"{t:%H:%M:%S}" for t in x_times]
            )
            if self.border_title:
                self.plt.ylim(lower=0, upper=100 if "%" in self.border_title else None)
        else:
            width, height = self.size.width or 60, self.size.height or 15
            self.plt.text(
                "Waiting for data...",
                width // 2,
                height // 2,
                alignment="center",
            )


class GPUMonitorApp(App):
    """A Textual app to monitor NVIDIA GPU status."""

    BINDINGS: ClassVar[list[BindingType]] = [
        ("q", "quit", "Quit"),
        ("l", "toggle_log", "Toggle Log"),
    ]
    TITLE = "GPUMonitor"
    CSS = """
    #info-panel {
        height: auto;
        padding: 0 1;
        border: round $primary;
        content-align: center middle;
        margin-bottom: 1;
    }
    #plots-grid {
        height: 1fr;
    }
    #raw_log {
        height: 7;
    }
    Grid {
        grid-size: 2 2;
        grid-gutter: 1;
    }
    GPUPlot, DataTable, #raw_log {
        border: round $primary;
    }
    DataTable {
        height: 100%; background: $surface;
    }
    DataTable > .datatable--header { background: $surface-darken-2; }
    DataTable > .datatable--odd-row { background: $surface; }
    DataTable > .datatable--even-row { background: $surface-darken-1; }
    """
    gpu_id: int
    info_panel: Static
    util_plot: GPUPlot
    mem_plot: GPUPlot
    power_plot: GPUPlot
    proc_table: DataTable
    raw_log: RichLog

    def __init__(self, gpu_id: int = 0) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.info_panel = Static(content="Querying GPU Info...", id="info-panel")
        self.util_plot = GPUPlot()
        self.mem_plot = GPUPlot()
        self.power_plot = GPUPlot()
        self.proc_table = DataTable()
        self.raw_log = RichLog(max_lines=100, id="raw_log", highlight=True, markup=True)

    def compose(self) -> ComposeResult:
        self.util_plot.border_title = f"GPU-{self.gpu_id} Utilization (%)"
        self.mem_plot.border_title = f"GPU-{self.gpu_id} Memory Util (%)"
        self.power_plot.border_title = f"GPU-{self.gpu_id} Power (W)"
        self.proc_table.border_title = f"GPU-{self.gpu_id} Processes"
        self.raw_log.border_title = f"GPU-{self.gpu_id} Raw dmon Output (L to Toggle)"
        yield Header()
        with Vertical():
            yield self.info_panel
            with Grid(id="plots-grid"):
                yield self.util_plot
                yield self.mem_plot
                yield self.power_plot
                yield self.proc_table
            yield self.raw_log
        yield Footer()

    def on_mount(self) -> None:
        self.proc_table.add_columns("PID", "Process Name", "GPU Memory")
        self.run_worker(
            work=self.update_info_panel, exclusive=True, group="initialization"
        )
        self.run_worker(work=self.poll_dmon_stats, exclusive=True, group="dmon_polling")
        self.set_interval(interval=DEFAULT_PROC_POLL, callback=self.update_process_list)

    def action_toggle_log(self) -> None:
        self.raw_log.display = not self.raw_log.display

    async def poll_dmon_stats(self) -> None:
        """Polls `nvidia-smi dmon` to update GPU metrics in real-time."""
        command: list[str] = [*DMON_BASE_CMD, str(self.gpu_id)]

        async with subprocess_lifespan(command, log=self.raw_log) as proc:
            if proc.stdout is None or proc.stderr is None:
                self.raw_log.write(content="[red]Failed to capture dmon output.[/red]")
                return
            stdout = proc.stdout
            stderr = proc.stderr

            async def consume_stdout() -> None:
                while not stdout.at_eof():
                    line: bytes = await stdout.readline()
                    if not line:
                        break
                    decoded: str = line.decode(
                        encoding="utf-8", errors="ignore"
                    ).strip()
                    if decoded.startswith("#"):
                        continue
                    self.raw_log.write(content=decoded)
                    parts: list[str] = re.split(pattern=r"\s+", string=decoded)
                    try:
                        p, u, m = float(parts[1]), int(parts[4]), int(parts[5])
                        self.power_plot.update_data(p)
                        self.util_plot.update_data(u)
                        self.mem_plot.update_data(m)
                    except (IndexError, ValueError):
                        pass

            async def consume_stderr() -> None:
                while not stderr.at_eof():
                    line: bytes = await stderr.readline()
                    if not line:
                        break
                    self.raw_log.write(
                        content="[bold red]DMON STDERR:[/bold red] "
                        f"{line.decode(encoding='utf-8', errors='ignore').strip()}"
                    )

            await asyncio.gather(consume_stdout(), consume_stderr())

    async def update_info_panel(self) -> None:
        """Query nvidia-smi for static GPU info to set info panel."""
        command: list[str] = [*INFO_BASE_CMD, str(self.gpu_id)]

        name, driver, mem, cuda = "N/A", "N/A", "N/A", "N/A"
        async with subprocess_lifespan(command, log=self.raw_log) as proc:
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(stderr.decode(encoding="utf-8", errors="ignore"))
            output: str = stdout.decode()
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
            if name_match:
                name: str = name_match.group(1).strip()
            if driver_match:
                driver: str = driver_match.group(1).strip()
            if cuda_match:
                cuda: str = cuda_match.group(1).strip()
            if mem_match:
                mem: str = mem_match.group(1).strip()
            info_text: str = (
                f"[green][b]GPU:[/b] {name} | [b]Driver:[/b] {driver} | "
                f"[b]CUDA:[/b] {cuda} | [b]Memory:[/b] {mem}[/]"
            )
            self.info_panel.update(content=info_text)

    async def update_process_list(self) -> None:
        """Queries for running processes and updates the table."""

        async with subprocess_lifespan(PROCESS_QUERY_CMD, log=self.raw_log) as proc:
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                self.raw_log.write(
                    "[red]Proc Query Failed:[/] "
                    f"{stderr.decode(encoding='utf-8', errors='ignore')}"
                )
                return
            self.proc_table.clear()
            output: str = stdout.decode(encoding="utf-8").strip()
            if not output or "[Not Supported]" in output:
                self.proc_table.add_row("[i]No running processes[/i]")
            else:
                for line in output.splitlines():
                    try:
                        pid, name, memory = line.split(",")
                        self.proc_table.add_row(
                            pid.strip(), name.strip(), f"{memory.strip()} MiB"
                        )
                    except ValueError:
                        self.raw_log.write(
                            f"[orange]WARN: Malformed process line:[/] {line}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monitor NVIDIA GPU metrics in real-time."
    )
    parser.add_argument(
        "gpu_id",
        type=int,
        nargs="?",
        default=0,
        help="ID of the GPU to monitor (default: 0)",
    )
    parser.add_argument(
        "-s",
        "--history_size",
        type=int,
        default=1,
        help="Number of historical data points to keep in plots "
        f"(default: {DEFAULT_HISTORY})",
    )
    parser.add_argument(
        "-d",
        "--dmon-poll-interval",
        type=int,
        default=DEFAULT_DMON_POLL,
        help="Interval in seconds between polling GPU stats "
        f"(default: {DEFAULT_DMON_POLL})",
    )
    parser.add_argument(
        "-p",
        "--proc-poll-interval",
        type=int,
        default=DEFAULT_DMON_POLL,
        help="Interval in seconds between polling GPU stats "
        f"(default: {DEFAULT_DMON_POLL})",
    )
    parser.add_argument(
        "-t",
        "--local-timezone",
        type=str,
        default=LOCAL_TIMEZONE,
        help=f"Timezone for displaying timestamps (default: {LOCAL_TIMEZONE})",
    )
    args = parser.parse_args()
    app = GPUMonitorApp(gpu_id=args.gpu_id)
    app.run()


if __name__ == "__main__":
    main()
