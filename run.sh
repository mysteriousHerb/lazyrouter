#!/bin/bash
cd /home/lazyrouter
uv run main.py


# use pm2 to run this script in the background and restart it if it crashes
# pm2 start run.sh --name lazyrouter
# add to startup: pm2 startup, then pm2 save