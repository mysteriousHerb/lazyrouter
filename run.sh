#!/bin/bash
# Get the directory where the script is located and change to it
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR" || { echo "Failed to cd to $SCRIPT_DIR" >&2; exit 1; }

# Run the app with explicit config and port options
uv run python main.py --config config.yaml --port 8000


# use pm2 to run this script in the background and restart it if it crashes
# pm2 start run.sh --name lazyrouter
# add to startup: pm2 startup, then pm2 save