"""Simple benchmark for dLoRA serving without trace files."""
import argparse
import asyncio
import json
import random
import time
from typing import List, Tuple

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from vllm.transformers_utils.tokenizer import get_tokenizer

REQUEST_LATENCY: List[Tuple[int, int, float]] = []
SENT_COUNT = 0
RECEIVED_COUNT = 0
SENT_MODEL_IDS: List[int] = []


def get_skewed_model_weights(num_models: int, skewness: float) -> np.ndarray:
    """Generate probability weights for model selection with skewness.
    
    Args:
        num_models: Number of LoRA models
        skewness: Skewness parameter (0 = uniform, higher = more skewed to lower IDs)
                  Uses Zipf-like distribution: P(i) ∝ 1/(i+1)^skewness
    
    Returns:
        Normalized probability weights for each model
    """
    if skewness == 0:
        return np.ones(num_models) / num_models
    weights = np.array([1.0 / (i + 1) ** skewness for i in range(num_models)])
    return weights / weights.sum()


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_len_min: int,
    input_len_max: int,
    output_len: int,
) -> List[Tuple[str, int, int]]:
    """Sample requests from the dataset.
    
    Args:
        dataset_path: Path to ShareGPT dataset
        num_requests: Number of requests to sample
        tokenizer: Tokenizer to use
        input_len_min: Minimum input token length (inclusive)
        input_len_max: Maximum input token length (inclusive)
        output_len: Fixed output token length for all requests
    """
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Filter conversations with at least 2 turns
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Tokenize
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    
    tokenized_dataset = []
    for i in range(len(dataset)):
        tokenized_dataset.append((prompts[i], prompt_token_ids[i]))

    # Filter based on input token length range
    filtered_dataset = []
    for prompt, tokens in tokenized_dataset:
        prompt_len = len(tokens)
        # Filter by user-specified input length range
        if prompt_len < input_len_min or prompt_len > input_len_max:
            continue
        # Use user-specified fixed output length
        filtered_dataset.append((prompt, prompt_len, output_len))

    if len(filtered_dataset) == 0:
        raise ValueError(f"No requests found with input length in range [{input_len_min}, {input_len_max}]. "
                         f"Try adjusting the range.")
    
    if len(filtered_dataset) < num_requests:
        print(f"Warning: Only {len(filtered_dataset)} requests available in range [{input_len_min}, {input_len_max}], "
              f"requested {num_requests}")

    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))
    
    prompt_lens = [t[1] for t in sampled_requests]
    print(f"Sampled {len(sampled_requests)} requests")
    print(f"Input length range: [{input_len_min}, {input_len_max}]")
    print(f"Actual input length: min={min(prompt_lens)}, max={max(prompt_lens)}, avg={sum(prompt_lens) / len(prompt_lens):.1f}")
    print(f"Output length (fixed): {output_len}")
    
    return sampled_requests


async def send_request(
    session: aiohttp.ClientSession,
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int,
    model_id: int,
) -> None:
    """Send a single request."""
    global RECEIVED_COUNT
    request_start_time = time.time()
    
    pload = {
        "prompt": prompt,
        "model_id": model_id,
        "n": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": False,
    }
    
    headers = {"User-Agent": "Benchmark Client"}
    timeout = aiohttp.ClientTimeout(total=3600)
    
    async with session.post(api_url, headers=headers, json=pload) as response:
        output = await response.text()
    
    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))
    RECEIVED_COUNT += 1


async def progress_printer(total_requests: int, interval: float = 2.0) -> None:
    """Periodically print progress of sent and received requests."""
    start_time = time.time()
    while RECEIVED_COUNT < total_requests:
        elapsed = time.time() - start_time
        print(f"[{elapsed:6.1f}s] Sent: {SENT_COUNT:4d}/{total_requests}  |  Received: {RECEIVED_COUNT:4d}/{total_requests}")
        await asyncio.sleep(interval)


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    num_models: int,
    request_rate: float,
    skewness: float,
) -> None:
    """Run the benchmark."""
    global SENT_COUNT
    tasks = []
    timeout = aiohttp.ClientTimeout(total=3600)
    total_requests = len(input_requests)
    
    # Compute model selection weights based on skewness
    model_weights = get_skewed_model_weights(num_models, skewness)
    model_ids = np.arange(num_models)
    
    # Start progress printer
    printer_task = asyncio.create_task(progress_printer(total_requests))
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, (prompt, prompt_len, output_len) in enumerate(input_requests):
            model_id = int(np.random.choice(model_ids, p=model_weights))
            task = asyncio.create_task(
                send_request(session, api_url, prompt, prompt_len, output_len, model_id)
            )
            tasks.append(task)
            SENT_COUNT += 1
            SENT_MODEL_IDS.append(model_id)
            
            if request_rate < float("inf"):
                interval = np.random.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)
        
        await asyncio.gather(*tasks)
    
    # Cancel the printer task once all requests are done
    printer_task.cancel()
    try:
        await printer_task
    except asyncio.CancelledError:
        pass
    
    # Print final status with total model ID distribution
    total_model_counts = {}
    for mid in SENT_MODEL_IDS:
        total_model_counts[mid] = total_model_counts.get(mid, 0) + 1
    total_model_str = ", ".join(f"{k}:{v}" for k, v in sorted(total_model_counts.items()))
    print(f"[DONE]   Sent: {SENT_COUNT:4d}/{total_requests}  |  Received: {RECEIVED_COUNT:4d}/{total_requests}")
    print(f"         Total LoRA ID distribution: {total_model_str}")


def reset_swap_stats(host: str, port: int) -> None:
    """Reset swap stats on the server."""
    import requests
    try:
        resp = requests.post(f"http://{host}:{port}/reset_swap_stats", timeout=10)
        if resp.status_code == 200:
            print("Swap stats reset on server")
    except Exception as e:
        print(f"Warning: Could not reset swap stats: {e}")


def get_swap_stats(host: str, port: int) -> dict:
    """Get swap stats from the server."""
    import requests
    try:
        resp = requests.get(f"http://{host}:{port}/swap_stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Warning: Could not get swap stats: {e}")
    return None


def main(args):
    print(f"Configuration: {args}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer)
    input_requests = sample_requests(
        args.dataset, 
        args.num_prompts, 
        tokenizer,
        args.input_len_min,
        args.input_len_max,
        args.output_len,
    )

    # Reset swap stats before benchmark
    reset_swap_stats(args.host, args.port)

    print(f"\nStarting benchmark with {len(input_requests)} requests...")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Number of LoRA models: {args.num_models}")
    print(f"LoRA skewness: {args.skewness} (0=uniform, higher=more skewed)")
    print()

    benchmark_start_time = time.time()
    asyncio.run(benchmark(api_url, input_requests, args.num_models, args.request_rate, args.skewness))
    benchmark_end_time = time.time()
    
    benchmark_time = benchmark_end_time - benchmark_start_time

    # Compute statistics
    latencies = [lat for _, _, lat in REQUEST_LATENCY]
    avg_latency = np.mean(latencies)
    p50_latency = np.percentile(latencies, 50)
    p90_latency = np.percentile(latencies, 90)
    p99_latency = np.percentile(latencies, 99)
    
    avg_per_output_token = np.mean([
        latency / output_len for _, output_len, latency in REQUEST_LATENCY
    ])

    # Get swap stats from server
    swap_stats = get_swap_stats(args.host, args.port)

    print("=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Total requests:        {len(REQUEST_LATENCY)}")
    print(f"Total time:            {benchmark_time:.2f} s")
    print(f"Throughput:            {len(REQUEST_LATENCY) / benchmark_time:.2f} req/s")
    print()
    print(f"Avg latency:           {avg_latency:.2f} s")
    print(f"P50 latency:           {p50_latency:.2f} s")
    print(f"P90 latency:           {p90_latency:.2f} s")
    print(f"P99 latency:           {p99_latency:.2f} s")
    print()
    print(f"Avg latency/output_token: {avg_per_output_token:.4f} s")
    print()
    if swap_stats:
        print("LoRA SWAP STATISTICS")
        print("-" * 30)
        print("Initialization:")
        print(f"  Total swap operations: {swap_stats['total']['init']['swap_calls']}")
        print(f"  Total models swapped:  {swap_stats['total']['init']['swap_count']}")
        print("Runtime:")
        print(f"  Total swap operations: {swap_stats['total']['runtime']['swap_calls']}")
        print(f"  Total models swapped:  {swap_stats['total']['runtime']['swap_count']}")
        print("-" * 30)
        print("Per-engine breakdown:")
        for engine_id, stats in swap_stats['per_engine'].items():
            print(f"  Engine {engine_id}:")
            print(f"    Init:    {stats['init']['swap_calls']} ops, {stats['init']['swap_count']} models")
            print(f"    Runtime: {stats['runtime']['swap_calls']} ops, {stats['runtime']['swap_count']} models")
    print("=" * 50)

    # Reset runtime swap stats at the end (init stats preserved)
    reset_swap_stats(args.host, args.port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple benchmark for dLoRA")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to ShareGPT dataset")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer name or path")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to process")
    parser.add_argument("--num-models", type=int, default=16,
                        help="Number of LoRA models")
    parser.add_argument("--request-rate", type=float, default=4.0,
                        help="Requests per second (use 'inf' for max)")
    parser.add_argument("--input-len-min", type=int, default=128,
                        help="Minimum input token length (inclusive)")
    parser.add_argument("--input-len-max", type=int, default=512,
                        help="Maximum input token length (inclusive)")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Fixed output token length for all requests")
    parser.add_argument("--skewness", type=float, default=2.0,
                        help="Skewness for LoRA model selection (0=uniform, higher=more skewed to lower IDs)")
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    main(args)

