import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from textual.widgets import RichLog


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
