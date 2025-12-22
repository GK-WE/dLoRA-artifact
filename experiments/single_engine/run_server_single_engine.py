#!/usr/bin/env python3
"""
Single vLLM Engine Server with Configurable LoRA HBM Budget.

This script runs a single vLLM engine for experiments measuring the impact of 
LoRA HBM budget on TTFT (Time To First Token).

Usage:
    python run_server_single_engine.py \
        --model meta-llama/Llama-2-7b-hf \
        --total-loras 128 \
        --gpu-loras 32 \
        --port 8000

Arguments:
    --model: Base model name/path (HuggingFace model)
    --total-loras: Total number of LoRA adapters available (CPU-side)
    --gpu-loras: Number of LoRA adapters to preload into GPU HBM
    --port: Server port (default: 8000)
"""

import argparse
import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.config import ExecType
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.ray_utils import ray, initialize_cluster
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.logger import init_logger

from ray.air.util.torch_dist import init_torch_dist_process_group

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds

app = FastAPI()
engine: Optional[AsyncLLMEngine] = None
engine_config: Dict = {}
gpu_loras_setting: int = 0


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "config": engine_config})


@app.get("/swap_stats")
async def get_swap_stats() -> JSONResponse:
    """Get LoRA swap statistics from the engine."""
    if engine is None:
        return JSONResponse({"error": "Engine not initialized"}, status_code=500)
    
    result = ray.get(engine.engine.get_swap_stats.remote())
    return JSONResponse({
        "init": {"swap_calls": result[0], "swap_count": result[1]},
        "runtime": {"swap_calls": result[2], "swap_count": result[3]}
    })


@app.post("/reset_swap_stats")
async def reset_swap_stats() -> JSONResponse:
    """Reset runtime LoRA swap statistics (init stats preserved)."""
    if engine is None:
        return JSONResponse({"error": "Engine not initialized"}, status_code=500)
    
    ray.get(engine.engine.reset_swap_stats.remote())
    return JSONResponse({"status": "ok", "message": "Runtime swap stats reset"})


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - model_id: the LoRA adapter ID to use (0 to total_loras-1).
    - stream: whether to stream the results or not (optional).
    - max_tokens: maximum tokens to generate (optional, default 16).
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    if engine is None:
        return JSONResponse({"error": "Engine not initialized"}, status_code=500)
    
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    model_id = int(request_dict.pop("model_id", 0))
    stream = request_dict.pop("stream", False)
    
    # Default max_tokens to small value for quick testing
    if "max_tokens" not in request_dict:
        request_dict["max_tokens"] = 16
    
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    
    # Record arrival time for TTFT measurement
    ttft_start = time.time()
    
    # Generate with the single engine
    results_generator = engine.generate(prompt, sampling_params, request_id, model_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        first_token_time = None
        async for request_output in results_generator:
            if first_token_time is None and len(request_output.outputs) > 0:
                if len(request_output.outputs[0].token_ids) > 0:
                    first_token_time = time.time()
            
            prompt_text = request_output.prompt
            text_outputs = [
                prompt_text + output.text for output in request_output.outputs
            ]
            ret = {
                "text": text_outputs,
                "finished": request_output.finished,
            }
            if first_token_time:
                ret["ttft"] = first_token_time - ttft_start
            yield (json.dumps(ret) + "\0").encode("utf-8")

    if stream:
        return StreamingResponse(stream_results())

    # Non-streaming case
    final_output = None
    first_token_time = None
    async for request_output in results_generator:
        if first_token_time is None and len(request_output.outputs) > 0:
            if len(request_output.outputs[0].token_ids) > 0:
                first_token_time = time.time()
        
        if await request.is_disconnected():
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt_text = final_output.prompt
    text_outputs = [prompt_text + output.text for output in final_output.outputs]
    
    ttft = first_token_time - ttft_start if first_token_time else None
    completion_time = time.time() - ttft_start
    
    ret = {
        "text": text_outputs,
        "model_id": model_id,
        "ttft": ttft,
        "completion_time": completion_time,
    }
    return JSONResponse(ret)


def create_single_engine(args) -> AsyncLLMEngine:
    """Create a single AsyncLLMEngine with the specified configuration."""
    global engine_config, gpu_loras_setting
    
    gpu_loras_setting = args.gpu_loras
    
    engine_config = {
        "model": args.model,
        "total_loras": args.total_loras,
        "gpu_loras": args.gpu_loras,
        "dtype": args.dtype,
        "max_num_seqs": args.max_num_seqs,
    }
    
    logger.info(f"Creating single engine with config: total_loras={args.total_loras}, gpu_loras={args.gpu_loras}")
    
    # Use AsyncEngineArgs to create the engine - matches the existing pattern
    engine_args = AsyncEngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        tokenizer_mode='auto',
        trust_remote_code=args.trust_remote_code,
        download_dir=args.download_dir,
        use_np_weights=False,
        use_dummy_weights=args.use_dummy_weights,
        dtype=args.dtype,
        seed=args.seed,
        worker_use_ray=True,
        pipeline_parallel_size=args.pipeline_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        block_size=args.block_size,
        swap_space=args.swap_space,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        policy=args.policy,
        exec_type=3,  # ExecType.LORA = 3
        num_groups=1,  # Single engine
        num_models=args.total_loras,  # Total LoRA adapters (CPU-side)
        num_model_per_group=args.total_loras,
        max_r=args.max_r,
        gpu_capacity=args.gpu_loras,  # LoRAs preloaded into GPU HBM
        disable_log_stats=False,
        engine_use_ray=True,
        disable_log_requests=False,
    )
    
    # Create the engine using from_engine_args
    async_engine = AsyncLLMEngine.from_engine_args(engine_args, engine_id=0, start_engine_loop=True)
    
    # Initialize torch distributed process group for the workers
    # This is required before calling _init_workers_ray_cont
    all_workers = async_engine.workers
    init_torch_dist_process_group(all_workers, backend="nccl")
    
    # Profile memory and initialize
    result = ray.get(async_engine.engine._init_workers_ray_cont.remote())
    available_gpu_memory, lora_weight_size, cache_block_size = result
    
    logger.info(f"Available GPU memory: {available_gpu_memory / (1 << 30):.2f} GB")
    logger.info(f"LoRA weight size per adapter: {lora_weight_size / (1 << 20):.2f} MB")
    
    # Determine which LoRA adapters to preload into GPU HBM
    # Preload the first `gpu_loras` adapters (0 to gpu_loras-1)
    init_gpu_lora_list = list(range(args.gpu_loras))
    
    # Calculate memory needed for GPU LoRA adapters
    # IMPORTANT: Must subtract LoRA memory from available memory before allocating KV cache
    lora_memory_needed = lora_weight_size * len(init_gpu_lora_list)
    cache_memory = available_gpu_memory - lora_memory_needed
    
    logger.info(f"LoRA memory needed for {len(init_gpu_lora_list)} adapters: {lora_memory_needed / (1 << 30):.2f} GB")
    logger.info(f"Memory available for KV cache: {cache_memory / (1 << 30):.2f} GB")
    
    if cache_memory <= 0:
        raise ValueError(f"Not enough GPU memory for {len(init_gpu_lora_list)} LoRA adapters. "
                         f"Required: {lora_memory_needed / (1 << 30):.2f} GB, "
                         f"Available: {available_gpu_memory / (1 << 30):.2f} GB. "
                         f"Try reducing --gpu-loras or using a smaller model.")
    
    logger.info(f"Preloading {args.gpu_loras} LoRA adapters into GPU HBM: {init_gpu_lora_list[:10]}{'...' if len(init_gpu_lora_list) > 10 else ''}")
    
    # Initialize cache and preload LoRA adapters
    ray.get(async_engine.engine.init_cont.remote(cache_memory, init_gpu_lora_list))
    
    # Get models in GPU for verification
    models_in_gpu = ray.get(async_engine.engine.get_models_in_gpu.remote())
    logger.info(f"LoRA adapters now in GPU HBM: {models_in_gpu[:10]}{'...' if len(models_in_gpu) > 10 else ''}")
    
    return async_engine


def main():
    parser = argparse.ArgumentParser(
        description="Single vLLM Engine Server with Configurable LoRA HBM Budget"
    )
    
    # Core experiment arguments
    parser.add_argument("--model", type=str, default="NousResearch/Llama-2-7b-hf",
                        help="Base model name or path")
    parser.add_argument("--total-loras", type=int, required=True,
                        help="Total number of LoRA adapters available (CPU-side)")
    parser.add_argument("--gpu-loras", type=int, required=True,
                        help="Number of LoRA adapters to preload into GPU HBM")
    parser.add_argument("--port", type=int, default=8000,
                        help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    
    # Model configuration
    parser.add_argument("--tokenizer", type=str, default="hf-internal-testing/llama-tokenizer",
                        help="Tokenizer name or path (defaults to model)")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code from HuggingFace")
    parser.add_argument("--download-dir", type=str, default=None,
                        help="Directory to download model weights")
    parser.add_argument("--use-dummy-weights", action="store_true", default=True,
                        help="Use dummy weights for model (faster startup)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "half", "bfloat16", "float"],
                        help="Data type for model weights")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    
    # Memory and cache configuration
    parser.add_argument("--block-size", type=int, default=16,
                        choices=[8, 16, 32],
                        help="Token block size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--swap-space", type=int, default=4,
                        help="CPU swap space size (GiB)")
    
    # Scheduling configuration
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096,
                        help="Maximum number of batched tokens per iteration")
    parser.add_argument("--max-num-seqs", type=int, default=16,
                        help="Maximum number of sequences per iteration")
    parser.add_argument("--policy", type=str, default="credit",
                        help="Scheduling policy")
    
    # Parallelism configuration
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1,
                        help="Number of pipeline stages")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1,
                        help="Number of tensor parallel replicas")
    
    # LoRA configuration
    parser.add_argument("--max-r", type=int, default=8,
                        help="Maximum LoRA rank")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.gpu_loras > args.total_loras:
        parser.error(f"gpu-loras ({args.gpu_loras}) cannot exceed total-loras ({args.total_loras})")
    if args.gpu_loras < 1:
        parser.error("gpu-loras must be at least 1")
    if args.total_loras < 1:
        parser.error("total-loras must be at least 1")
    
    global engine
    engine = create_single_engine(args)
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Configuration: total_loras={args.total_loras}, gpu_loras={args.gpu_loras}")
    logger.info(f"Adapters 0-{args.gpu_loras-1} are preloaded in GPU HBM")
    logger.info(f"Adapters {args.gpu_loras}-{args.total_loras-1} require swap from CPU")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )


if __name__ == "__main__":
    main()
