#!/bin/bash

# =============================================================================
# IMPORTANT: Run this script with "source" to keep the environment variables:
#   source benchmarks/cleanup_server.sh
# OR:
#   . benchmarks/cleanup_server.sh
#
# Running as "bash cleanup_server.sh" will NOT work because env vars won't persist!
# =============================================================================

# Stop Ray
ray stop --force 2>/dev/null

# Kill all related processes
pkill -9 -f "vllm" 2>/dev/null
pkill -9 -f "ray::" 2>/dev/null  
pkill -9 -f "raylet" 2>/dev/null
pkill -9 -f "gcs_server" 2>/dev/null

# Clean NCCL shared memory (IMPORTANT!)
rm -rf /dev/shm/nccl-* 2>/dev/null
rm -rf /dev/shm/*ray* 2>/dev/null

# Clean Ray temp files  
rm -rf /tmp/ray/* 2>/dev/null

# Wait for GPU memory release
sleep 3

# Set NCCL environment variables to prevent hangs (CRITICAL!)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1      # Prevents PCIe P2P communication hangs
export NCCL_IB_DISABLE=1       # Disables InfiniBand (not available on this system)
export NCCL_SOCKET_IFNAME="^lo,docker"

echo ""
echo "============================================"
echo "Cleanup complete!"
echo ""
echo "Environment variables set:"
echo "  NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
echo "  NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
echo ""
echo "Server initialization takes ~40-50 seconds."
echo "============================================"
