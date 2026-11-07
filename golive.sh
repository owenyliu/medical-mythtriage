#!/bin/bash
cd "$(dirname "$0")"
export RUMOR_AGENT_ENV_FILE=~/assignment1-harness/.env
export MYTHTRIAGE_ENV_FILE=~/assignment1-harness/.env
exec python3 server.py --live
