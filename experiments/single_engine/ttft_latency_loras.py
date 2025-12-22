#!/usr/bin/env python3
"""
TTFT/Latency Measurement Script for LoRA HBM Budget Experiments

This script measures TTFT and latency for a single vLLM engine with a specific
LoRA HBM budget configuration. Use after selecting arrival rate with probe_arrival_rate.py.

Usage:
    python ttft_latency_loras.py \
        --endpoint http://localhost:8000 \
        --dataset /path/to/ShareGPT.json \
        --tokenizer hf-internal-testing/llama-tokenizer \
        --total-loras 128 \
        --gpu-loras 32 \
        --prompt-regime short \
        --arrival-rate 2.0 \
        --num-requests 200 \
        --out results.csv

Workflow:
    1. Start server: python run_server_single_engine.py --total-loras 128 --gpu-loras 32 ...
    2. Run measurement: python ttft_latency_loras.py --total-loras 128 --gpu-loras 32 ...
    3. Repeat for different gpu-loras values
"""

import argparse
import asyncio
import aiohttp
import json
import random
import time
import sys
import statistics
import datetime
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


# Prompt regimes: (min_tokens, max_tokens)
PROMPT_REGIMES = {
    "short": (256, 1024),
    "long": (2048, 4096),
}


@dataclass
class RequestResult:
    """Result of a single request."""
    request_id: int
    model_id: int
    prompt_tokens: int
    start_time: float
    end_time: float
    ttft: Optional[float]
    latency: float
    status: str
    error_message: Optional[str] = None


def load_and_filter_prompts(
    dataset_path: str,
    tokenizer,
    prompt_regime: str,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """Load prompts from ShareGPT dataset and filter by token length."""
    min_tokens, max_tokens = PROMPT_REGIMES[prompt_regime]
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Filter conversations with at least 1 turn
    dataset = [data for data in dataset if len(data.get("conversations", [])) >= 1]
    
    # Extract prompts (first human turn)
    prompts = []
    for data in dataset:
        for conv in data["conversations"]:
            if conv.get("from") == "human":
                prompts.append(conv["value"])
                break
    
    print(f"Found {len(prompts)} prompts in dataset")
    
    # Tokenize and filter by length
    print(f"Filtering for {prompt_regime} regime ({min_tokens}-{max_tokens} tokens)...")
    filtered_prompts = []
    
    batch_size = 1000
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokenized = tokenizer(batch, truncation=False, add_special_tokens=True)
        
        for j, tokens in enumerate(tokenized.input_ids):
            token_count = len(tokens)
            if min_tokens <= token_count <= max_tokens:
                filtered_prompts.append((batch[j], token_count))
    
    print(f"Found {len(filtered_prompts)} prompts in {prompt_regime} regime")
    
    if len(filtered_prompts) == 0:
        raise ValueError(f"No prompts found with {min_tokens}-{max_tokens} tokens.")
    
    # Shuffle with seed
    rng = random.Random(seed)
    rng.shuffle(filtered_prompts)
    
    token_counts = [t for _, t in filtered_prompts]
    print(f"Token stats: min={min(token_counts)}, max={max(token_counts)}, "
          f"mean={sum(token_counts)/len(token_counts):.1f}")
    
    return filtered_prompts


async def send_request(
    session: aiohttp.ClientSession,
    endpoint: str,
    request_id: int,
    model_id: int,
    prompt: str,
    prompt_tokens: int,
    timeout: float,
    max_tokens: int = 16,
) -> RequestResult:
    """Send a single request and measure timing."""
    
    url = f"{endpoint}/generate"
    payload = {
        "prompt": prompt,
        "model_id": model_id,
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    start_time = time.time()
    ttft = None
    status = "success"
    error_message = None
    
    try:
        async with session.post(
            url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            if response.status != 200:
                status = "error"
                error_message = f"HTTP {response.status}"
            else:
                first_chunk_received = False
                async for chunk in response.content.iter_any():
                    if not first_chunk_received:
                        ttft = time.time() - start_time
                        first_chunk_received = True
                    
                    try:
                        for part in chunk.decode('utf-8').split('\0'):
                            if part.strip():
                                data = json.loads(part)
                                if "ttft" in data and data["ttft"] is not None:
                                    ttft = data["ttft"]
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
                        
    except asyncio.TimeoutError:
        status = "timeout"
        error_message = f"Request timed out after {timeout}s"
    except aiohttp.ClientError as e:
        status = "error"
        error_message = str(e)
    except Exception as e:
        status = "error"
        error_message = f"Unexpected error: {e}"
    
    end_time = time.time()
    latency = end_time - start_time
    
    if ttft is None and status == "success":
        ttft = latency
    
    return RequestResult(
        request_id=request_id,
        model_id=model_id,
        prompt_tokens=prompt_tokens,
        start_time=start_time,
        end_time=end_time,
        ttft=ttft,
        latency=latency,
        status=status,
        error_message=error_message,
    )


async def run_measurement(
    endpoint: str,
    prompts: List[Tuple[str, int]],
    total_loras: int,
    gpu_loras: int,
    prompt_regime: str,
    arrival_rate: float,
    num_requests: Optional[int],
    duration_seconds: Optional[float],
    timeout: float,
    max_tokens: int = 16,
    seed: int = 42,
) -> List[RequestResult]:
    """Run the TTFT/latency measurement.
    
    Either num_requests or duration_seconds should be specified.
    If both are specified, num_requests takes precedence.
    """
    
    tasks: List[asyncio.Task] = []
    inter_arrival = 1.0 / arrival_rate if arrival_rate > 0 else 0
    
    # Determine mode
    use_duration = num_requests is None and duration_seconds is not None
    if use_duration:
        expected_requests = int(duration_seconds * arrival_rate)
        mode_str = f"{duration_seconds}s duration (~{expected_requests} requests)"
    else:
        expected_requests = num_requests
        mode_str = f"{num_requests} requests"
    
    # Token stats
    sample_size = min(expected_requests, len(prompts))
    token_counts = [t for _, t in prompts[:sample_size]]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    
    print(f"\n{'='*60}")
    print(f"TTFT/Latency Measurement")
    print(f"{'='*60}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Total LoRAs: {total_loras}")
    print(f"  GPU LoRAs: {gpu_loras}")
    print(f"  Prompt regime: {prompt_regime}")
    print(f"  Arrival rate: {arrival_rate} rps")
    print(f"  Mode: {mode_str}")
    print(f"  Avg prompt tokens: {avg_tokens:.1f}")
    print(f"  Timeout: {timeout}s")
    print(f"{'='*60}\n")
    
    rng = random.Random(seed)
    
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()
        request_id = 0
        
        # Determine stop condition
        if use_duration:
            def should_continue():
                return time.time() - start_time < duration_seconds
        else:
            def should_continue():
                return request_id < num_requests
        
        while should_continue():
            prompt_idx = request_id % len(prompts)
            prompt, prompt_tokens = prompts[prompt_idx]
            model_id = rng.randint(0, total_loras - 1)
            
            task = asyncio.create_task(
                send_request(
                    session=session,
                    endpoint=endpoint,
                    request_id=request_id,
                    model_id=model_id,
                    prompt=prompt,
                    prompt_tokens=prompt_tokens,
                    timeout=timeout,
                    max_tokens=max_tokens,
                )
            )
            tasks.append(task)
            request_id += 1
            
            if request_id % 20 == 0:
                elapsed = time.time() - start_time
                if use_duration:
                    print(f"  Sent {request_id} requests ({elapsed:.1f}s / {duration_seconds}s)...", end='\r')
                else:
                    print(f"  Sent {request_id}/{num_requests} requests ({elapsed:.1f}s)...", end='\r')
            
            # Wait for next arrival
            if inter_arrival > 0:
                next_arrival = start_time + request_id * inter_arrival
                sleep_time = next_arrival - time.time()
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        print(f"\n  Waiting for {len(tasks)} requests to complete...")
        results = await asyncio.gather(*tasks)
    
    return results


def compute_percentile(values: List[float], p: float) -> float:
    """Compute percentile of a list of values."""
    if not values:
        return float('nan')
    sorted_values = sorted(values)
    idx = int(len(sorted_values) * p / 100)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


def compute_statistics(results: List[RequestResult], total_loras: int, gpu_loras: int) -> dict:
    """Compute summary statistics from results."""
    
    total = len(results)
    if total == 0:
        return {"error": "No results"}
    
    successes = [r for r in results if r.status == "success"]
    errors = [r for r in results if r.status == "error"]
    timeouts = [r for r in results if r.status == "timeout"]
    
    success_count = len(successes)
    error_count = len(errors)
    timeout_count = len(timeouts)
    
    ttfts = [r.ttft for r in successes if r.ttft is not None]
    latencies = [r.latency for r in successes]
    prompt_tokens_list = [r.prompt_tokens for r in successes]
    
    stats = {
        "total_loras": total_loras,
        "gpu_loras": gpu_loras,
        "total_requests": total,
        "successful": success_count,
        "errors": error_count,
        "timeouts": timeout_count,
        "success_rate": success_count / total * 100 if total > 0 else 0,
        "error_rate": (error_count + timeout_count) / total * 100 if total > 0 else 0,
    }
    
    if prompt_tokens_list:
        stats.update({
            "prompt_tokens_min": min(prompt_tokens_list),
            "prompt_tokens_max": max(prompt_tokens_list),
            "prompt_tokens_mean": statistics.mean(prompt_tokens_list),
        })
    
    if ttfts:
        stats.update({
            "ttft_min": min(ttfts),
            "ttft_median": statistics.median(ttfts),
            "ttft_mean": statistics.mean(ttfts),
            "ttft_p90": compute_percentile(ttfts, 90),
            "ttft_p99": compute_percentile(ttfts, 99),
            "ttft_max": max(ttfts),
        })
    
    if latencies:
        stats.update({
            "latency_min": min(latencies),
            "latency_median": statistics.median(latencies),
            "latency_mean": statistics.mean(latencies),
            "latency_p90": compute_percentile(latencies, 90),
            "latency_p99": compute_percentile(latencies, 99),
            "latency_max": max(latencies),
        })
    
    return stats


def print_summary(stats: dict):
    """Print a formatted summary of the results."""
    
    print(f"\n{'='*60}")
    print(f"RESULTS: total_loras={stats.get('total_loras')}, gpu_loras={stats.get('gpu_loras')}")
    print(f"{'='*60}")
    
    print(f"\nRequest counts:")
    print(f"  Total:      {stats.get('total_requests', 0)}")
    print(f"  Successful: {stats.get('successful', 0)}")
    print(f"  Errors:     {stats.get('errors', 0)}")
    print(f"  Timeouts:   {stats.get('timeouts', 0)}")
    print(f"  Error rate: {stats.get('error_rate', 0):.2f}%")
    
    if "prompt_tokens_mean" in stats:
        print(f"\nPrompt tokens:")
        print(f"  Min:  {stats['prompt_tokens_min']}")
        print(f"  Max:  {stats['prompt_tokens_max']}")
        print(f"  Mean: {stats['prompt_tokens_mean']:.1f}")
    
    if "ttft_median" in stats:
        print(f"\nTTFT (Time to First Token):")
        print(f"  Min:    {stats['ttft_min']*1000:.1f} ms")
        print(f"  Median: {stats['ttft_median']*1000:.1f} ms")
        print(f"  Mean:   {stats['ttft_mean']*1000:.1f} ms")
        print(f"  P90:    {stats['ttft_p90']*1000:.1f} ms")
        print(f"  P99:    {stats['ttft_p99']*1000:.1f} ms")
        print(f"  Max:    {stats['ttft_max']*1000:.1f} ms")
    
    if "latency_median" in stats:
        print(f"\nLatency (end-to-end):")
        print(f"  Min:    {stats['latency_min']*1000:.1f} ms")
        print(f"  Median: {stats['latency_median']*1000:.1f} ms")
        print(f"  Mean:   {stats['latency_mean']*1000:.1f} ms")
        print(f"  P90:    {stats['latency_p90']*1000:.1f} ms")
        print(f"  P99:    {stats['latency_p99']*1000:.1f} ms")
        print(f"  Max:    {stats['latency_max']*1000:.1f} ms")
    
    print(f"{'='*60}\n")


def save_results(
    stats: dict,
    output_path: str,
    args: argparse.Namespace,
):
    """Save results to file (appends if exists)."""
    
    path = Path(output_path)
    file_exists = path.exists()
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add metadata to stats
    stats["run_timestamp"] = run_timestamp
    stats["prompt_regime"] = args.prompt_regime
    stats["arrival_rate"] = args.arrival_rate
    stats["num_requests_arg"] = args.num_requests  # The argument (may be None if using duration)
    stats["duration_seconds"] = args.duration_seconds  # The duration argument (may be None)
    
    if path.suffix.lower() == '.csv':
        import csv
        fieldnames = [
            "run_timestamp", "total_loras", "gpu_loras", "prompt_regime", "arrival_rate",
            "duration_seconds", "total_requests", "successful", "errors", "timeouts",
            "success_rate", "error_rate",
            "prompt_tokens_min", "prompt_tokens_max", "prompt_tokens_mean",
            "ttft_min", "ttft_median", "ttft_mean", "ttft_p90", "ttft_p99", "ttft_max",
            "latency_min", "latency_median", "latency_mean", "latency_p90", "latency_p99", "latency_max",
        ]
        
        mode = 'a' if file_exists else 'w'
        with open(path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(stats)
    else:
        # JSONL
        mode = 'a' if file_exists else 'w'
        with open(path, mode) as f:
            record = {"type": "measurement", **stats}
            f.write(json.dumps(record) + '\n')
    
    action = "appended to" if file_exists else "saved to"
    print(f"Results {action}: {path}")


async def check_server_health(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if the server is healthy."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/health",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"Server healthy: {data}")
                    return True
    except Exception as e:
        print(f"Server health check failed: {e}")
    return False


async def get_swap_stats(endpoint: str) -> Optional[dict]:
    """Get swap stats from the server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{endpoint}/swap_stats",
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status == 200:
                    return await response.json()
    except Exception:
        pass
    return None


async def reset_swap_stats(endpoint: str) -> bool:
    """Reset swap stats on the server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{endpoint}/reset_swap_stats",
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                return response.status == 200
    except Exception:
        pass
    return False


async def main():
    parser = argparse.ArgumentParser(
        description="TTFT/Latency Measurement for LoRA HBM Budget Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example workflow:
    # Start server with specific gpu_loras
    python run_server_single_engine.py --total-loras 128 --gpu-loras 32 ...
    
    # Run measurement
    python ttft_latency_loras.py \\
        --dataset ShareGPT.json --tokenizer llama-tokenizer \\
        --total-loras 128 --gpu-loras 32 \\
        --prompt-regime short --arrival-rate 2.0 \\
        --num-requests 200 --out results.csv
    
    # Restart server with different gpu_loras and repeat
        """
    )
    
    parser.add_argument("--endpoint", "-e", type=str, default="http://localhost:8000",
                        help="Server endpoint URL")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT dataset JSON file")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer name or path")
    parser.add_argument("--total-loras", type=int, required=True,
                        help="Total number of LoRA adapters (CPU-side)")
    parser.add_argument("--gpu-loras", type=int, required=True,
                        help="Number of LoRA adapters in GPU HBM (for recording)")
    parser.add_argument("--prompt-regime", type=str, choices=["short", "long"], default="short",
                        help="Prompt length regime")
    parser.add_argument("--arrival-rate", "-r", type=float, default=2.0,
                        help="Request arrival rate (requests/second)")
    parser.add_argument("--num-requests", "-n", type=int, default=None,
                        help="Number of requests to send (use this OR --duration-seconds)")
    parser.add_argument("--duration-seconds", "-d", type=float, default=None,
                        help="Duration in seconds (use this OR --num-requests)")
    parser.add_argument("--timeout", "-t", type=float, default=120.0,
                        help="Request timeout in seconds")
    parser.add_argument("--out", "-o", type=str, default=None,
                        help="Output file (CSV or JSONL, appends if exists)")
    parser.add_argument("--max-tokens", type=int, default=16,
                        help="Maximum output tokens per request")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--skip-health-check", action="store_true",
                        help="Skip server health check")
    parser.add_argument("--show-swap-stats", action="store_true",
                        help="Show LoRA swap statistics after measurement")
    
    args = parser.parse_args()
    
    # Validate: need either num_requests or duration_seconds
    if args.num_requests is None and args.duration_seconds is None:
        parser.error("Must specify either --num-requests or --duration-seconds")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Load prompts
    prompts = load_and_filter_prompts(
        args.dataset,
        tokenizer,
        args.prompt_regime,
        seed=args.seed,
    )
    
    # Check if we have enough prompts
    if args.num_requests is not None:
        expected_requests = args.num_requests
    else:
        expected_requests = int(args.duration_seconds * args.arrival_rate)
    
    if len(prompts) < expected_requests:
        print(f"Warning: Only {len(prompts)} prompts available, will cycle through them")
    
    # Check server health
    if not args.skip_health_check:
        print("Checking server health...")
        healthy = await check_server_health(args.endpoint)
        if not healthy:
            print("ERROR: Server is not healthy. Use --skip-health-check to bypass.")
            sys.exit(1)
    
    # Reset swap stats before measurement
    print("Resetting swap stats...")
    await reset_swap_stats(args.endpoint)
    
    # Run measurement
    results = await run_measurement(
        endpoint=args.endpoint,
        prompts=prompts,
        total_loras=args.total_loras,
        gpu_loras=args.gpu_loras,
        prompt_regime=args.prompt_regime,
        arrival_rate=args.arrival_rate,
        num_requests=args.num_requests,
        duration_seconds=args.duration_seconds,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    
    # Compute and print statistics
    stats = compute_statistics(results, args.total_loras, args.gpu_loras)
    print_summary(stats)
    
    # Show swap stats if requested
    if args.show_swap_stats:
        swap_stats = await get_swap_stats(args.endpoint)
        if swap_stats:
            print("LoRA Swap Statistics:")
            print(f"  Init:    {swap_stats.get('init', {})}")
            print(f"  Runtime: {swap_stats.get('runtime', {})}")
    
    # Save results
    if args.out:
        save_results(stats, args.out, args)


if __name__ == "__main__":
    asyncio.run(main())

