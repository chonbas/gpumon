#!/usr/bin/bash

# Entrypoint wrapper for `sysmon` Python package for use from a standard shell.
# Uses `uv` to create a virtual environment if one does not already exist.
# Launches main `sysmon` entrypoint within the virtual environment, allowing
# for easy integration into `/usr/local/bin` or similar PATH.

set -e

# Resolve symlinks to find the actual repository root
SOURCE="${BASH_SOURCE[0]}"
while [ -L "$SOURCE" ]; do
    DIR="$(cd "$(dirname "$SOURCE")" && pwd)"
    SOURCE="$(readlink "$SOURCE")"
    # Handle relative symlinks
    [[ "$SOURCE" != /* ]] && SOURCE="$DIR/$SOURCE"
done
REPO_ROOT="$(cd "$(dirname "$SOURCE")" && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it from https://docs.astral.sh/uv/"
    exit 1
fi

# Create virtual environment if it does not exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment..."
    uv venv "${VENV_DIR}" --python 3.12
fi

# Sync dependencies from pyproject.toml
uv sync --project "${REPO_ROOT}" --quiet

# Run sysmon
uv run --project "${REPO_ROOT}" sysmon "$@"

