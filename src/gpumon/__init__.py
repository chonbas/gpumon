import argparse

from ._defaults import (
    DEFAULT_DMON_POLL,
    DEFAULT_HISTORY,
    DEFAULT_PROC_POLL,
    LOCAL_TIMEZONE,
)
from ._monitor import SystemMonitor


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
        default=DEFAULT_PROC_POLL,
        help="Interval in seconds between polling GPU stats "
        f"(default: {DEFAULT_PROC_POLL})",
    )
    parser.add_argument(
        "-t",
        "--local-timezone",
        type=str,
        default=LOCAL_TIMEZONE,
        help=f"Timezone for displaying timestamps (default: {LOCAL_TIMEZONE})",
    )
    args = parser.parse_args()
    app = SystemMonitor(gpu_id=args.gpu_id)
    app.run()


if __name__ == "__main__":
    main()
