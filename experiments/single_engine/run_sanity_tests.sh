#!/bin/bash
# Sanity tests for Single Engine LoRA HBM Experiment
#
# This script runs the server with different configurations and tests each one.
# Run from the dLoRA-artifact directory with the virtual environment activated.
#
# Usage:
#   cd /home/cc/dLoRA-artifact
#   source lora-env/bin/activate
#   bash experiments/single_engine/run_sanity_tests.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$PROJECT_DIR"

# Configuration
MODEL="${MODEL:-facebook/opt-125m}"
BASE_PORT="${BASE_PORT:-8000}"

echo "=============================================="
echo "Single Engine LoRA HBM Sanity Tests"
echo "=============================================="
echo "Model: $MODEL"
echo "Base port: $BASE_PORT"
echo ""

# Test configurations: (total_loras, gpu_loras)
declare -a TEST_CONFIGS=(
    "128 2"
    "128 128"
    "256 2"
    "256 256"
)

run_test() {
    local total_loras=$1
    local gpu_loras=$2
    local port=$3
    
    echo ""
    echo "=============================================="
    echo "Test: total_loras=$total_loras, gpu_loras=$gpu_loras"
    echo "=============================================="
    
    # Start server in background
    echo "Starting server on port $port..."
    python experiments/single_engine/run_server_single_engine.py \
        --model "$MODEL" \
        --total-loras "$total_loras" \
        --gpu-loras "$gpu_loras" \
        --port "$port" \
        --use-dummy-weights &
    
    SERVER_PID=$!
    
    # Wait for server to start
    echo "Waiting for server to be ready..."
    sleep 10
    
    for i in {1..30}; do
        if curl -s "http://localhost:$port/health" > /dev/null 2>&1; then
            echo "Server is ready!"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "ERROR: Server did not start in time"
            kill $SERVER_PID 2>/dev/null || true
            return 1
        fi
        sleep 2
    done
    
    # Run smoke test
    echo "Running smoke test..."
    if python experiments/single_engine/smoke_test_server.py \
        --port "$port" \
        --total-loras "$total_loras" \
        --gpu-loras "$gpu_loras"; then
        echo "✓ Test PASSED"
        RESULT=0
    else
        echo "✗ Test FAILED"
        RESULT=1
    fi
    
    # Stop server
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    
    # Give time for cleanup
    sleep 2
    
    return $RESULT
}

# Run all tests
FAILED=0
PORT=$BASE_PORT

for config in "${TEST_CONFIGS[@]}"; do
    read -r total_loras gpu_loras <<< "$config"
    
    if ! run_test "$total_loras" "$gpu_loras" "$PORT"; then
        FAILED=$((FAILED + 1))
    fi
    
    PORT=$((PORT + 1))
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All tests PASSED"
    exit 0
else
    echo "✗ $FAILED test(s) FAILED"
    exit 1
fi

