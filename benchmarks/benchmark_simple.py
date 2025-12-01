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


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    """Sample requests from the dataset."""
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
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long/short sequences
    filtered_dataset = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    sampled_requests = random.sample(filtered_dataset, min(num_requests, len(filtered_dataset)))
    
    prompt_lens = [t[1] for t in sampled_requests]
    output_lens = [t[2] for t in sampled_requests]
    print(f"Sampled {len(sampled_requests)} requests")
    print(f"avg_prompt_len: {sum(prompt_lens) / len(prompt_lens):.1f}")
    print(f"avg_output_len: {sum(output_lens) / len(output_lens):.1f}")
    
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


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    num_models: int,
    request_rate: float,
) -> None:
    """Run the benchmark."""
    tasks = []
    timeout = aiohttp.ClientTimeout(total=3600)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for i, (prompt, prompt_len, output_len) in enumerate(input_requests):
            model_id = random.randint(0, num_models - 1)
            task = asyncio.create_task(
                send_request(session, api_url, prompt, prompt_len, output_len, model_id)
            )
            tasks.append(task)
            
            if request_rate < float("inf"):
                interval = np.random.exponential(1.0 / request_rate)
                await asyncio.sleep(interval)
        
        await asyncio.gather(*tasks)


def main(args):
    print(f"Configuration: {args}")
    random.seed(args.seed)
    np.random.seed(args.seed)

    api_url = f"http://{args.host}:{args.port}/generate"
    tokenizer = get_tokenizer(args.tokenizer)
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    print(f"\nStarting benchmark with {len(input_requests)} requests...")
    print(f"Request rate: {args.request_rate} req/s")
    print(f"Number of LoRA models: {args.num_models}")
    print()

    benchmark_start_time = time.time()
    asyncio.run(benchmark(api_url, input_requests, args.num_models, args.request_rate))
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
    print("=" * 50)


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
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    main(args)

