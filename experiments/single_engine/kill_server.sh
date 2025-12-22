#!/bin/bash

PATTERN="experiments/single_engine/run_server_single_engine.py"

# Find matching PIDs (exclude the grep process itself)
PIDS=$(ps -ef | grep "$PATTERN" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No running process found for: $PATTERN"
    exit 0
fi

echo "Found process(es): $PIDS"
echo "Killing..."

kill -9 $PIDS

echo "Done."
