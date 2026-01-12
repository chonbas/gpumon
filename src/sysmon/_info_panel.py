import asyncio
import getpass
import os
import re

import psutil
from textual.widgets import RichLog, Static

from sysmon._plots import DataPlot, memory_formatter
from sysmon._utils import GIB, MIB, subprocess_lifespan

INFO_BASE_CMD: list[str] = ["nvidia-smi", "-q", "-i"]


async def update_info_panel(
    info_panel: Static,
    log: RichLog,
    mem_plot: DataPlot,
    power_plot: DataPlot,
) -> None:
    """Updates the info panel with CPU, OS, and GPU information. Sets default
    values in memory and power plots based on GPU specs.
    Args:
        info_panel: The Static widget to update
        log: RichLog widget for logging errors
        mem_plot: Memory plot to set formatter for
        power_plot: Power plot to set y-axis limit
    """
    info_panel.update(content="Querying info...")
    sys_info: str = ""
    nv_info: str = ""
    n_gpus = await _get_cuda_count(log=log)
    gpu_infos = await asyncio.gather(
        asyncio.to_thread(_get_cpu_os_info),
        *[
            _get_gpu_info(
                log=log,
                mem_plot=mem_plot,
                power_plot=power_plot,
                gpu_id=gpu_id,
            )
            for gpu_id in range(n_gpus)
        ],
    )
    for gpu_info in gpu_infos:
        if not isinstance(gpu_info, dict):
            sys_info = gpu_info
            continue
        for info_text in gpu_info.values():
            nv_info += f"{info_text}\n"

    nv_info = nv_info.strip()
    info_panel.update(content=f"{sys_info}\n{nv_info}")


def _get_cpu_os_info() -> str:
    """Helper to parse CPU and OS info using psutil and os.uname()."""
    os_info = os.uname()
    arch: str = os_info.machine

    # Try to get CPU name from /proc/cpuinfo on Linux
    cpu_name: str = "N/A"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass

    cpu_cores: int | None = psutil.cpu_count(logical=False)
    logical_cpus: int | None = psutil.cpu_count(logical=True)
    node: str = os.uname().nodename
    release: str = os_info.release
    sysname: str = os_info.sysname
    total_ram: float = psutil.virtual_memory().total / GIB
    user: str = getpass.getuser()
    return (
        f"{user} | {node} | {sysname} | {release} | {arch}\n"
        f"CPU: {cpu_name} | Cores: {cpu_cores or 'N/A'} | "
        f"Logical CPUs: {logical_cpus or 'N/A'} | "
        f"Total RAM:{total_ram: .2f} GiB"
    )


async def _get_cuda_count(log: RichLog) -> int:
    """Helper to get the number of CUDA-capable GPUs. Tries to use PyTorch if available,
    otherwise falls back to parsing nvidia-smi output.
    Args:
        log: RichLog widget for logging errors
    Returns:
        Number of CUDA-capable GPUs detected
    """
    try:
        # Try to just use PyTorch if available
        import torch  # type: ignore[import-error]

        await asyncio.sleep(0)  # Ensure function is async
        return torch.cuda.device_count()
    except ImportError:
        # Fallback to nvidia-smi parsing
        async with subprocess_lifespan(
            ["nvidia-smi", "--list-gpus"],
            log=log,
            name="cuda-count",
        ) as proc:
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return 0
            output = stdout.decode()
            match = re.search(r"GPU\s+(\d+):", output)
            if match:
                return int(match.group(1)) + 1
            return 0


async def _get_gpu_info(
    log: RichLog,
    mem_plot: DataPlot,
    power_plot: DataPlot,
    gpu_id: int = 0,
) -> dict[int, str]:
    """Parse GPU info from nvidia-smi output and set formatters.
    Args:
        output: Raw nvidia-smi -q output
        mem_plot: Memory plot to set formatter for
        power_plot: Power plot to set y-axis limit
        gpu_id: GPU index for setting per-series formatter
    """
    command: list[str] = [*INFO_BASE_CMD, str(gpu_id)]
    async with subprocess_lifespan(command, log=log, name="info-panel") as proc:
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(stderr.decode(encoding="utf-8", errors="ignore"))

        output: str = stdout.decode(encoding="utf-8", errors="ignore")

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
    info_text: str = (
        f"GPU-{gpu_id}: {name} | Driver: {driver} | CUDA: {cuda} | VRAM: {mem}"
    )
    return {gpu_id: info_text}
