#!/usr/bin/env python3
"""
Arrival Rate Selection Helper

This script helps determine a reasonable arrival rate for experiments by:
1. Sending requests at a specified rate for a duration
2. Measuring TTFT, latency, and error rates
3. Outputting per-request logs and summary statistics

Uses real prompts from ShareGPT dataset with accurate token counting via tokenizer.

Usage:
    python probe_arrival_rate.py \
        --endpoint http://localhost:8000 \
        --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --tokenizer hf-internal-testing/llama-tokenizer \
        --total-loras 128 \
        --prompt-regime short \
        --arrival-rate 2.0 \
        --duration-seconds 60 \
        --out results.jsonl

Selection Rule for "Middle" Arrival Rate:
-----------------------------------------
A good "middle" arrival rate should satisfy:
  1. Error/timeout rate < 1%
  2. TTFT shows some variation (system slightly stressed, not idle)
  3. TTFT is not exploding (p99 < 5x median, no runaway queue growth)
  4. P90 TTFT is reasonable (< 2-3 seconds for most workloads)

Recommended testing approach:
  1. Start with low rate (0.5 rps) to establish baseline
  2. Increase gradually: 0.5, 1, 2, 4, 8, 12, 16 rps
  3. Stop when error rate > 1% or TTFT explodes
  4. Choose the highest rate that meets all criteria
"""

import argparse
import asyncio
import aiohttp
import json
import random
import time
import sys
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path


# Default arrival rates to test (requests per second)
DEFAULT_ARRIVAL_RATES = [0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 16.0]

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
    ttft: Optional[float]  # Time to first token (seconds)
    latency: float  # Total request latency (seconds)
    status: str  # "success", "error", "timeout"
    error_message: Optional[str] = None
    output_tokens: int = 0


def load_and_filter_prompts(
    dataset_path: str,
    tokenizer,
    prompt_regime: str,
    seed: int = 42,
) -> List[Tuple[str, int]]:
    """
    Load prompts from ShareGPT dataset and filter by token length.
    
    Args:
        dataset_path: Path to ShareGPT JSON file
        tokenizer: HuggingFace tokenizer
        prompt_regime: "short" or "long"
        seed: Random seed for shuffling
    
    Returns:
        List of (prompt_text, token_count) tuples
    """
    min_tokens, max_tokens = PROMPT_REGIMES[prompt_regime]
    
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Filter conversations with at least 1 turn
    dataset = [
        data for data in dataset
        if len(data.get("conversations", [])) >= 1
    ]
    
    # Extract prompts (first human turn)
    prompts = []
    for data in dataset:
        for conv in data["conversations"]:
            if conv.get("from") == "human":
                prompts.append(conv["value"])
                break
    
    print(f"Found {len(prompts)} prompts in dataset")
    
    # Tokenize and filter by length
    print(f"Tokenizing and filtering for {prompt_regime} regime ({min_tokens}-{max_tokens} tokens)...")
    filtered_prompts = []
    
    # Process in batches for efficiency
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
        raise ValueError(
            f"No prompts found with {min_tokens}-{max_tokens} tokens. "
            f"Check your dataset or try a different regime."
        )
    
    # Shuffle with seed for reproducibility
    rng = random.Random(seed)
    rng.shuffle(filtered_prompts)
    
    # Report statistics
    token_counts = [t for _, t in filtered_prompts]
    print(f"Token count stats: min={min(token_counts)}, max={max(token_counts)}, "
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
        "stream": True,  # Use streaming for accurate TTFT
    }
    
    start_time = time.time()
    ttft = None
    status = "success"
    error_message = None
    output_tokens = 0
    
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
                # Stream the response to get TTFT
                first_chunk_received = False
                async for chunk in response.content.iter_any():
                    if not first_chunk_received:
                        ttft = time.time() - start_time
                        first_chunk_received = True
                    
                    # Try to parse chunk for server-reported TTFT
                    try:
                        # Handle null-terminated JSON chunks
                        for part in chunk.decode('utf-8').split('\0'):
                            if part.strip():
                                data = json.loads(part)
                                if "ttft" in data and data["ttft"] is not None:
                                    # Use server-reported TTFT if available
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
    
    # If no TTFT was measured (non-streaming or error), use latency as fallback
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
        output_tokens=output_tokens,
    )


async def run_load_test(
    endpoint: str,
    prompts: List[Tuple[str, int]],
    total_loras: int,
    prompt_regime: str,
    arrival_rate: float,
    duration_seconds: float,
    timeout: float,
    max_tokens: int = 16,
    seed: int = 42,
) -> List[RequestResult]:
    """Run the load test at specified arrival rate."""
    
    results: List[RequestResult] = []
    tasks: List[asyncio.Task] = []
    
    # Calculate inter-arrival time
    inter_arrival = 1.0 / arrival_rate if arrival_rate > 0 else float('inf')
    
    # Total expected requests
    expected_requests = int(duration_seconds * arrival_rate)
    
    # Get token stats for prompts we'll use
    num_prompts_available = len(prompts)
    token_counts = [t for _, t in prompts[:expected_requests]]
    if token_counts:
        avg_tokens = sum(token_counts) / len(token_counts)
    else:
        avg_tokens = 0
    
    print(f"\n{'='*60}")
    print(f"Starting load test:")
    print(f"  Endpoint: {endpoint}")
    print(f"  Arrival rate: {arrival_rate} rps")
    print(f"  Duration: {duration_seconds}s")
    print(f"  Expected requests: {expected_requests}")
    print(f"  Prompt regime: {prompt_regime}")
    print(f"  Prompts available: {num_prompts_available}")
    print(f"  Avg prompt tokens: {avg_tokens:.1f}")
    print(f"  Total LoRAs: {total_loras}")
    print(f"  Timeout: {timeout}s")
    print(f"{'='*60}\n")
    
    # RNG for model selection
    rng = random.Random(seed)
    
    connector = aiohttp.TCPConnector(limit=0)  # No connection limit
    async with aiohttp.ClientSession(connector=connector) as session:
        start_time = time.time()
        request_id = 0
        
        while time.time() - start_time < duration_seconds:
            # Get prompt (cycle through if we run out)
            prompt_idx = request_id % len(prompts)
            prompt, prompt_tokens = prompts[prompt_idx]
            
            # Random model selection
            model_id = rng.randint(0, total_loras - 1)
            
            # Schedule the request
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
            
            # Progress indicator
            if request_id % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Sent {request_id} requests ({elapsed:.1f}s elapsed)...", end='\r')
            
            # Wait for next arrival
            next_arrival = start_time + request_id * inter_arrival
            sleep_time = next_arrival - time.time()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        print(f"\n  Waiting for {len(tasks)} requests to complete...")
        
        # Wait for all requests to complete
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


def compute_statistics(results: List[RequestResult]) -> dict:
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
    
    # TTFT statistics (only for successful requests)
    ttfts = [r.ttft for r in successes if r.ttft is not None]
    latencies = [r.latency for r in successes]
    prompt_tokens_list = [r.prompt_tokens for r in successes]
    
    stats = {
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


def print_summary(stats: dict, arrival_rate: float, prompt_regime: str):
    """Print a formatted summary of the results."""
    
    print(f"\n{'='*60}")
    print(f"SUMMARY (arrival_rate={arrival_rate} rps, regime={prompt_regime})")
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
    
    # Arrival rate assessment
    print(f"\n{'='*60}")
    print("ASSESSMENT:")
    
    error_rate = stats.get('error_rate', 100)
    ttft_median = stats.get('ttft_median', float('inf'))
    ttft_p99 = stats.get('ttft_p99', float('inf'))
    
    issues = []
    
    if error_rate > 1.0:
        issues.append(f"❌ Error rate too high ({error_rate:.1f}% > 1%)")
    else:
        print(f"  ✓ Error rate OK ({error_rate:.2f}% < 1%)")
    
    if ttft_p99 > 5 * ttft_median and ttft_median > 0:
        ratio = ttft_p99 / ttft_median
        issues.append(f"❌ TTFT variance too high (p99/median = {ratio:.1f}x > 5x)")
    elif "ttft_median" in stats:
        ratio = ttft_p99 / ttft_median if ttft_median > 0 else 0
        print(f"  ✓ TTFT variance OK (p99/median = {ratio:.1f}x)")
    
    if ttft_p99 > 10.0:  # 10 seconds
        issues.append(f"❌ TTFT p99 too high ({ttft_p99*1000:.0f}ms > 10s)")
    elif "ttft_p99" in stats:
        print(f"  ✓ TTFT p99 reasonable ({ttft_p99*1000:.0f}ms)")
    
    if issues:
        print("\n  Issues found:")
        for issue in issues:
            print(f"    {issue}")
        print(f"\n  → This arrival rate ({arrival_rate} rps) may be TOO HIGH")
    else:
        print(f"\n  → This arrival rate ({arrival_rate} rps) looks ACCEPTABLE")
    
    print(f"{'='*60}\n")


def save_all_summaries(
    all_summaries: List[dict],
    output_path: str,
    args: argparse.Namespace,
):
    """Save all test summaries to a single file (JSONL or CSV).
    
    If the file exists, results are appended with a separator line.
    """
    import datetime
    
    path = Path(output_path)
    file_exists = path.exists()
    
    # Generate run timestamp for separator
    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if path.suffix.lower() == '.csv':
        import csv
        if all_summaries:
            # Define CSV columns in a logical order
            fieldnames = [
                "run_timestamp", "arrival_rate", "prompt_regime", "total_requests", "successful", 
                "errors", "timeouts", "success_rate", "error_rate",
                "prompt_tokens_min", "prompt_tokens_max", "prompt_tokens_mean",
                "ttft_min", "ttft_median", "ttft_mean", "ttft_p90", "ttft_p99", "ttft_max",
                "latency_min", "latency_median", "latency_mean", "latency_p90", "latency_p99", "latency_max",
                "status"
            ]
            
            # Append mode if file exists, write mode otherwise
            mode = 'a' if file_exists else 'w'
            with open(path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                
                if not file_exists:
                    # Write header only for new files
                    writer.writeheader()
                else:
                    # Add separator line for appending
                    f.write(f"# --- New run: {run_timestamp} ---\n")
                
                # Add timestamp to each row
                for s in all_summaries:
                    s["run_timestamp"] = run_timestamp
                    writer.writerow(s)
    else:
        # Default to JSONL
        mode = 'a' if file_exists else 'w'
        with open(path, mode) as f:
            if file_exists:
                # Add separator line for appending
                f.write(f'\n# === New run: {run_timestamp} ===\n')
            
            # Write metadata header
            metadata = {
                "type": "metadata",
                "run_timestamp": run_timestamp,
                "prompt_regime": args.prompt_regime,
                "total_loras": args.total_loras,
                "duration_seconds": args.duration_seconds,
                "endpoint": args.endpoint,
                "dataset": args.dataset,
                "tokenizer": args.tokenizer,
                "arrival_rates_tested": [s.get("arrival_rate") for s in all_summaries],
            }
            f.write(json.dumps(metadata) + '\n')
            
            # Write each summary
            for s in all_summaries:
                s["run_timestamp"] = run_timestamp
                record = {"type": "summary", **s}
                f.write(json.dumps(record) + '\n')
    
    action = "appended to" if file_exists else "saved to"
    print(f"\nResults {action}: {path}")


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


async def main():
    parser = argparse.ArgumentParser(
        description="Arrival Rate Selection Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Selection Rule for "Middle" Arrival Rate:
-----------------------------------------
A good "middle" arrival rate should satisfy:
  1. Error/timeout rate < 1%
  2. TTFT shows some variation (system slightly stressed, not idle)
  3. TTFT is not exploding (p99 < 5x median, no runaway queue growth)
  4. P90 TTFT is reasonable (< 2-3 seconds for most workloads)

Example workflow:
  # Test different arrival rates
  python probe_arrival_rate.py \\
      --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \\
      --tokenizer hf-internal-testing/llama-tokenizer \\
      --prompt-regime short \\
      --arrival-rates "0.5,1,2,4,8" \\
      --out results.jsonl
        """
    )
    
    parser.add_argument(
        "--endpoint", "-e",
        type=str,
        default="http://localhost:8000",
        help="Server endpoint URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to ShareGPT dataset JSON file"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer name or path (e.g., hf-internal-testing/llama-tokenizer)"
    )
    parser.add_argument(
        "--total-loras",
        type=int,
        default=128,
        help="Total number of LoRA adapters (default: 128)"
    )
    parser.add_argument(
        "--prompt-regime",
        type=str,
        choices=["short", "long"],
        default="short",
        help="Prompt length regime: short (256-1024 tokens) or long (2048-4096 tokens)"
    )
    parser.add_argument(
        "--arrival-rate", "-r",
        type=float,
        default=1.0,
        help="Request arrival rate in requests/second (default: 1.0)"
    )
    parser.add_argument(
        "--arrival-rates",
        type=str,
        default=None,
        help="Comma-separated list of arrival rates to test (e.g., '0.5,1,2,4,8'). "
             "If specified, runs multiple tests sequentially."
    )
    parser.add_argument(
        "--duration-seconds", "-d",
        type=float,
        default=60.0,
        help="Test duration in seconds (default: 60)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=120.0,
        help="Request timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default=None,
        help="Output file path (CSV or JSONL). If testing multiple rates, "
             "rate is appended to filename."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum tokens to generate per request (default: 16)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip server health check"
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    from vllm.transformers_utils.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(args.tokenizer)
    
    # Load and filter prompts
    prompts = load_and_filter_prompts(
        args.dataset,
        tokenizer,
        args.prompt_regime,
        seed=args.seed,
    )
    
    # Determine arrival rates to test
    if args.arrival_rates:
        rates = [float(r.strip()) for r in args.arrival_rates.split(',')]
    else:
        rates = [args.arrival_rate]
    
    # Check server health
    if not args.skip_health_check:
        print("Checking server health...")
        healthy = await check_server_health(args.endpoint)
        if not healthy:
            print("ERROR: Server is not healthy. Use --skip-health-check to bypass.")
            sys.exit(1)
    
    # Run tests for each arrival rate
    all_summaries = []
    
    for rate in rates:
        results = await run_load_test(
            endpoint=args.endpoint,
            prompts=prompts,
            total_loras=args.total_loras,
            prompt_regime=args.prompt_regime,
            arrival_rate=rate,
            duration_seconds=args.duration_seconds,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        
        stats = compute_statistics(results)
        stats["arrival_rate"] = rate
        stats["prompt_regime"] = args.prompt_regime
        
        # Determine status for this rate
        error_rate = stats.get('error_rate', 100)
        ttft_median = stats.get('ttft_median', float('inf'))
        ttft_p99 = stats.get('ttft_p99', float('inf'))
        ttft_ratio = ttft_p99 / ttft_median if ttft_median > 0 else float('inf')
        
        if error_rate > 1:
            stats["status"] = "TOO HIGH"
        elif ttft_ratio > 5:
            stats["status"] = "UNSTABLE"
        elif ttft_p99 > 10.0:  # 10 seconds
            stats["status"] = "SLOW"
        else:
            stats["status"] = "OK"
        
        all_summaries.append(stats)
        
        print_summary(stats, rate, args.prompt_regime)
        
        # Brief pause between tests
        if len(rates) > 1 and rate != rates[-1]:
            print("Pausing 5 seconds before next test...")
            await asyncio.sleep(5)
    
    # Print comparative summary if multiple rates tested
    if len(rates) > 1:
        print(f"\n{'='*80}")
        print(f"COMPARATIVE SUMMARY ({args.prompt_regime} prompts)")
        print(f"{'='*80}")
        print(f"{'Rate':>8} | {'Success%':>8} | {'Tokens':>8} | {'TTFT Med':>10} | {'TTFT P99':>10} | {'Lat Med':>10} | {'Status':>12}")
        print(f"{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
        
        for s in all_summaries:
            rate = s.get('arrival_rate', 0)
            success = s.get('success_rate', 0)
            avg_tokens = s.get('prompt_tokens_mean', 0)
            ttft_med = s.get('ttft_median', float('nan')) * 1000
            ttft_p99 = s.get('ttft_p99', float('nan')) * 1000
            lat_med = s.get('latency_median', float('nan')) * 1000
            status = s.get('status', 'UNKNOWN')
            
            print(f"{rate:>8.1f} | {success:>7.1f}% | {avg_tokens:>8.0f} | {ttft_med:>8.1f}ms | {ttft_p99:>8.1f}ms | {lat_med:>8.1f}ms | {status:>12}")
        
        print(f"{'='*80}")
        print("\nRecommendation: Choose the highest 'OK' rate for your experiments.")
        print("If all rates show 'OK', consider testing higher rates.")
        print("If low rates already show issues, consider reducing load or scaling up.")
    
    # Save all summaries to a single file
    if args.out:
        save_all_summaries(all_summaries, args.out, args)


if __name__ == "__main__":
    asyncio.run(main())
