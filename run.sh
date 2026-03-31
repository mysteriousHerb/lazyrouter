#!/bin/bash
# Get the absolute directory where the script is located and change to it
# This ensures we are always in the repository root before running uv
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || { echo "CRITICAL: Could not change to $SCRIPT_DIR" >&2; exit 1; }

# Log the working directory for debugging
echo "Starting lazyrouter in: $(pwd)"

# Run the app with uv
uv run python main.py


# use pm2 to run this script in the background and restart it if it crashes
# pm2 start run.sh --name lazyrouter
# add to startup: pm2 startup, then pm2 save