import asyncio
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from enum import Enum
from typing import TypeVar

from textual.widgets import RichLog

PLOT_HISTORY_SIZE = 100
LOCAL_TIMEZONE: str = os.getenv("LOCAL_TIMEZONE", default="US/Arizona")
POLL_INTERVAL = 1.0

KIB = 1024
MIB = 1024**2
GIB = 1024**3


E = TypeVar("E", bound=Enum)


def enum_from_value(enum: type[E], value: object, /) -> E:
    """Validate and coerce a value into an enum instance.
    Args:
        enum: The enum class to create an instance of.
        value: The value to convert to an enum instance.
    Returns:
        An instance of the enum class.
    Raises:
        ValueError: If the value cannot be converted to an enum instance.
    """
    try:
        if isinstance(value, enum):
            return value
        elif isinstance(value, str):
            str_val: str = value.removeprefix(".").replace("-", "_").strip()
            if str_val in enum:
                return enum(str_val)
            elif str_val.upper() in enum:
                return enum(str_val.upper())
            elif str_val.lower() in enum:
                return enum(str_val.lower())
            try:
                return enum[str_val.upper()]
            except KeyError:
                return enum[str_val.lower()]
        return enum(value)
    except (KeyError, ValueError, TypeError):
        raise ValueError(
            f"Unsupported {enum.__name__} value: {value!r} - choices "
            f"are: {list(enum.__members__)}"
        ) from None


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
    """Run a subprocess command and return its output.
    Args:
        command: The command to execute as a list of strings
        log: RichLog widget for logging output and errors
        name: Name of the subprocess for logging
        timeout: Timeout in seconds for the subprocess to complete
    Returns:
        The standard output of the subprocess as a string, or None on failure
    """
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
