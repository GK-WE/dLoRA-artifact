#!/usr/bin/env python3
"""
Smoke Test for Single Engine LoRA HBM Experiment Server.

This script tests that the server correctly handles requests for both:
- LoRA adapters that are preloaded in GPU HBM
- LoRA adapters that need to be swapped from CPU

Usage:
    # First start the server (in another terminal):
    python run_server_single_engine.py --model facebook/opt-125m --total-loras 128 --gpu-loras 2 --port 8000
    
    # Then run this test:
    python smoke_test_server.py --port 8000 --total-loras 128 --gpu-loras 2

Test configurations to validate:
    1. total_loras=128, gpu_loras=2
    2. total_loras=128, gpu_loras=128
    3. total_loras=256, gpu_loras=2
    4. total_loras=256, gpu_loras=256
"""

import argparse
import json
import requests
import time
import sys
from typing import Optional, Dict, Any


def make_request(
    base_url: str,
    prompt: str,
    model_id: int,
    max_tokens: int = 8,
    timeout: float = 60.0
) -> Dict[str, Any]:
    """Make a generate request to the server."""
    response = requests.post(
        f"{base_url}/generate",
        json={
            "prompt": prompt,
            "model_id": model_id,
            "max_tokens": max_tokens,
            "temperature": 0.0,  # Deterministic for testing
        },
        timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def check_health(base_url: str) -> Dict[str, Any]:
    """Check server health and get configuration."""
    response = requests.get(f"{base_url}/health", timeout=5.0)
    response.raise_for_status()
    return response.json()


def get_swap_stats(base_url: str) -> Dict[str, Any]:
    """Get LoRA swap statistics."""
    response = requests.get(f"{base_url}/swap_stats", timeout=5.0)
    response.raise_for_status()
    return response.json()


def reset_swap_stats(base_url: str) -> Dict[str, Any]:
    """Reset runtime swap statistics."""
    response = requests.post(f"{base_url}/reset_swap_stats", timeout=5.0)
    response.raise_for_status()
    return response.json()


def run_smoke_test(
    base_url: str,
    total_loras: int,
    gpu_loras: int,
    verbose: bool = True
) -> bool:
    """
    Run smoke test for a given configuration.
    
    Tests:
    1. Health check
    2. Request to adapter in HBM (should be fast, no swap)
    3. Request to adapter NOT in HBM (triggers swap)
    4. Verify swap stats
    
    Returns True if all tests pass.
    """
    print(f"\n{'='*60}")
    print(f"Smoke Test: total_loras={total_loras}, gpu_loras={gpu_loras}")
    print(f"{'='*60}")
    
    passed = True
    
    # Test 1: Health check
    print("\n[Test 1] Health check...")
    try:
        health = check_health(base_url)
        print(f"  ✓ Server is healthy")
        print(f"    Config: {health.get('config', {})}")
        
        # Verify configuration matches
        server_config = health.get('config', {})
        if server_config.get('total_loras') != total_loras:
            print(f"  ⚠ Warning: Server total_loras ({server_config.get('total_loras')}) != expected ({total_loras})")
        if server_config.get('gpu_loras') != gpu_loras:
            print(f"  ⚠ Warning: Server gpu_loras ({server_config.get('gpu_loras')}) != expected ({gpu_loras})")
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
        return False
    
    # Reset swap stats before testing
    print("\n[Test 2] Reset swap stats...")
    try:
        reset_swap_stats(base_url)
        print("  ✓ Swap stats reset")
    except Exception as e:
        print(f"  ⚠ Could not reset stats: {e}")
    
    # Test 3: Request to adapter that IS in HBM (adapter 0)
    print("\n[Test 3] Request to adapter IN GPU HBM (model_id=0)...")
    try:
        # Adapter 0 should always be in HBM if gpu_loras >= 1
        result = make_request(base_url, "Hello, how are you?", model_id=0, max_tokens=8)
        print(f"  ✓ Response received")
        print(f"    TTFT: {result.get('ttft', 'N/A'):.4f}s" if result.get('ttft') else "    TTFT: N/A")
        print(f"    Completion time: {result.get('completion_time', 'N/A'):.4f}s" if result.get('completion_time') else "    Completion time: N/A")
        print(f"    Model ID: {result.get('model_id')}")
        if verbose:
            text = result.get('text', [''])[0]
            print(f"    Output (truncated): {text[:100]}...")
    except Exception as e:
        print(f"  ✗ Request failed: {e}")
        passed = False
    
    # Test 4: Request to adapter that is NOT in HBM (if gpu_loras < total_loras)
    if gpu_loras < total_loras:
        # Pick an adapter that's definitely not in HBM
        out_of_hbm_id = total_loras - 1  # Last adapter, least likely to be preloaded
        print(f"\n[Test 4] Request to adapter NOT in GPU HBM (model_id={out_of_hbm_id})...")
        try:
            result = make_request(base_url, "The capital of France is", model_id=out_of_hbm_id, max_tokens=8)
            print(f"  ✓ Response received (swap from CPU should have occurred)")
            print(f"    TTFT: {result.get('ttft', 'N/A'):.4f}s" if result.get('ttft') else "    TTFT: N/A")
            print(f"    Completion time: {result.get('completion_time', 'N/A'):.4f}s" if result.get('completion_time') else "    Completion time: N/A")
            print(f"    Model ID: {result.get('model_id')}")
        except Exception as e:
            print(f"  ✗ Request failed: {e}")
            passed = False
    else:
        print(f"\n[Test 4] Skipped (gpu_loras == total_loras, all adapters in HBM)")
    
    # Test 5: Check swap stats
    print("\n[Test 5] Check swap statistics...")
    try:
        stats = get_swap_stats(base_url)
        print(f"  Init swaps: {stats.get('init', {}).get('swap_count', 'N/A')} models in {stats.get('init', {}).get('swap_calls', 'N/A')} calls")
        print(f"  Runtime swaps: {stats.get('runtime', {}).get('swap_count', 'N/A')} models in {stats.get('runtime', {}).get('swap_calls', 'N/A')} calls")
        
        # Verify runtime swap occurred if we tested out-of-HBM adapter
        if gpu_loras < total_loras:
            runtime_swaps = stats.get('runtime', {}).get('swap_count', 0)
            if runtime_swaps > 0:
                print(f"  ✓ Runtime swap detected as expected (adapter was not in HBM)")
            else:
                print(f"  ⚠ No runtime swaps detected (might be expected if adapter was already loaded)")
    except Exception as e:
        print(f"  ⚠ Could not get swap stats: {e}")
    
    # Test 6: Multiple requests to different adapters
    print("\n[Test 6] Multiple requests to test adapter diversity...")
    try:
        # Test a few adapters
        test_adapters = [0]  # Always test 0 (in HBM)
        if gpu_loras > 1:
            test_adapters.append(gpu_loras - 1)  # Last adapter in HBM
        if gpu_loras < total_loras:
            test_adapters.append(gpu_loras)  # First adapter NOT in HBM
        
        for adapter_id in test_adapters:
            in_hbm = adapter_id < gpu_loras
            try:
                result = make_request(base_url, f"Test prompt for adapter {adapter_id}", model_id=adapter_id, max_tokens=4)
                ttft = result.get('ttft')
                status = "IN HBM" if in_hbm else "NOT in HBM"
                ttft_str = f"{ttft:.4f}s" if ttft else "N/A"
                print(f"    Adapter {adapter_id} ({status}): TTFT={ttft_str}")
            except Exception as e:
                print(f"    Adapter {adapter_id}: FAILED - {e}")
                passed = False
    except Exception as e:
        print(f"  ✗ Multiple requests test failed: {e}")
        passed = False
    
    # Summary
    print(f"\n{'='*60}")
    if passed:
        print("✓ All smoke tests PASSED")
    else:
        print("✗ Some smoke tests FAILED")
    print(f"{'='*60}\n")
    
    return passed


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for Single Engine LoRA HBM Experiment Server"
    )
    parser.add_argument("--host", type=str, default="localhost",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--total-loras", type=int, required=True,
                        help="Total number of LoRA adapters (must match server)")
    parser.add_argument("--gpu-loras", type=int, required=True,
                        help="Number of LoRA adapters in GPU HBM (must match server)")
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    print(f"Testing server at {base_url}")
    print(f"Expected configuration: total_loras={args.total_loras}, gpu_loras={args.gpu_loras}")
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            check_health(base_url)
            print("Server is ready!")
            break
        except Exception:
            if i < max_retries - 1:
                time.sleep(1)
            else:
                print("Server not responding after 30 seconds. Is it running?")
                sys.exit(1)
    
    # Run smoke test
    success = run_smoke_test(
        base_url,
        args.total_loras,
        args.gpu_loras,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

