# Single Engine LoRA HBM Experiment

This directory contains scripts for measuring the impact of LoRA HBM (GPU memory) budget on TTFT (Time To First Token) in a **single vLLM engine**.

## Purpose

Before optimizing multi-engine/multi-GPU deployments, we need to validate that:
1. The number of LoRA adapters resident in GPU HBM affects TTFT
2. Swapping LoRA adapters from CPU to GPU introduces measurable latency

This experiment setup allows manual testing with different:
- Total LoRA adapter counts (128 or 256)
- GPU HBM budgets (number of adapters preloaded)
- Prompt length regimes (short: 256-1024 tokens, long: 2048-4096 tokens)

## Files

- `run_server_single_engine.py` - Server script that runs a single vLLM engine
- `smoke_test_server.py` - Sanity test script to verify the server works
- `probe_arrival_rate.py` - Arrival rate selection helper for experiments
- `ttft_latency_loras.py` - TTFT/latency measurement for LoRA HBM budget experiments

## Quick Start

### 1. Start the Server

```bash
# Activate the environment
cd /home/cc/dLoRA-artifact
source lora-env/bin/activate

# Start with a small configuration for testing
python experiments/single_engine/run_server_single_engine.py \
    --model facebook/opt-125m \
    --total-loras 128 \
    --gpu-loras 2 \
    --port 8000 \
    --use-dummy-weights
```

### 2. Run Smoke Test (in another terminal)

```bash
cd /home/cc/dLoRA-artifact
source lora-env/bin/activate

python experiments/single_engine/smoke_test_server.py \
    --port 8000 \
    --total-loras 128 \
    --gpu-loras 2
```

## Server Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Base model name/path (HuggingFace) | `facebook/opt-125m` |
| `--total-loras` | Total LoRA adapters available (CPU-side) | **Required** |
| `--gpu-loras` | LoRA adapters preloaded into GPU HBM | **Required** |
| `--port` | Server port | `8000` |
| `--use-dummy-weights` | Use dummy weights (faster startup) | `True` |
| `--dtype` | Data type (`auto`, `half`, `bfloat16`, `float`) | `auto` |
| `--gpu-memory-utilization` | Fraction of GPU memory to use | `0.90` |
| `--max-num-seqs` | Max sequences per iteration | `16` |

## Experiment Configurations

### Test Configurations for Sanity Check

```bash
# Config 1: total=128, gpu=2 (minimal HBM)
python run_server_single_engine.py --model facebook/opt-125m --total-loras 128 --gpu-loras 2 --port 8000

# Config 2: total=128, gpu=128 (all in HBM)
python run_server_single_engine.py --model facebook/opt-125m --total-loras 128 --gpu-loras 128 --port 8001

# Config 3: total=256, gpu=2 (minimal HBM, larger universe)
python run_server_single_engine.py --model facebook/opt-125m --total-loras 256 --gpu-loras 2 --port 8002

# Config 4: total=256, gpu=256 (all in HBM, larger universe)
python run_server_single_engine.py --model facebook/opt-125m --total-loras 256 --gpu-loras 256 --port 8003
```

### GPU HBM Budget Sweep (for TTFT measurement)

For `total_loras=128`:
```bash
for gpu_loras in 128 64 32 16 8 4 2; do
    echo "Testing gpu_loras=$gpu_loras"
    python run_server_single_engine.py \
        --model meta-llama/Llama-2-7b-hf \
        --total-loras 128 \
        --gpu-loras $gpu_loras \
        --port 8000
done
```

For `total_loras=256`:
```bash
for gpu_loras in 256 128 64 32 16 8 4 2; do
    echo "Testing gpu_loras=$gpu_loras"
    python run_server_single_engine.py \
        --model meta-llama/Llama-2-7b-hf \
        --total-loras 256 \
        --gpu-loras $gpu_loras \
        --port 8000
done
```

## API Endpoints

### POST /generate

Generate text with a specific LoRA adapter.

```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "Hello, how are you?",
        "model_id": 5,
        "max_tokens": 16
    }'
```

Response:
```json
{
    "text": ["Hello, how are you? I am doing well..."],
    "model_id": 5,
    "ttft": 0.0234,
    "completion_time": 0.156
}
```

### GET /health

Check server health and configuration.

```bash
curl http://localhost:8000/health
```

### GET /swap_stats

Get LoRA swap statistics (init vs runtime).

```bash
curl http://localhost:8000/swap_stats
```

Response:
```json
{
    "init": {"swap_calls": 1, "swap_count": 2},
    "runtime": {"swap_calls": 3, "swap_count": 5}
}
```

### POST /reset_swap_stats

Reset runtime swap counters (init stats preserved).

```bash
curl -X POST http://localhost:8000/reset_swap_stats
```

## Understanding Adapter Placement

When the server starts with `--total-loras N --gpu-loras M`:

- Adapters `0` to `M-1` are **preloaded** into GPU HBM at startup
- Adapters `M` to `N-1` are **only on CPU** and require swap when accessed

Example with `--total-loras 128 --gpu-loras 8`:
- `model_id=0` to `model_id=7` → in GPU HBM (fast)
- `model_id=8` to `model_id=127` → on CPU (requires swap, adds latency)

## Measuring TTFT

The `/generate` endpoint returns:
- `ttft`: Time To First Token (seconds) - time from request arrival to first token generated
- `completion_time`: Total completion time (seconds)

To measure the impact of HBM budget on TTFT:
1. Start server with specific `gpu_loras` setting
2. Reset swap stats: `POST /reset_swap_stats`
3. Send requests targeting adapters both IN and OUT of HBM
4. Record TTFT values from responses
5. Check swap stats: `GET /swap_stats` to verify swaps occurred

## Tips

1. **Use dummy weights for initial testing** - Much faster startup
2. **Reset swap stats between test runs** - Ensures clean measurements
3. **Test both in-HBM and out-of-HBM adapters** - To see the swap latency impact
4. **Use deterministic settings** - Set `temperature=0.0` for reproducible outputs

---

## Arrival Rate Selection

Use `probe_arrival_rate.py` to find a suitable arrival rate for experiments.

### Selection Rule for "Middle" Arrival Rate

A good arrival rate should satisfy:

1. **Error/timeout rate < 1%** - System is not overloaded
2. **TTFT shows variation** - System is slightly stressed (not idle)
3. **TTFT is not exploding** - p99 < 5× median (no runaway queue)
4. **P99 TTFT is reasonable** - Typically < 5-10 seconds

### Quick Start

```bash
# Test a single arrival rate
python probe_arrival_rate.py \
    --endpoint http://localhost:8000 \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --total-loras 128 \
    --prompt-regime short \
    --arrival-rate 2.0 \
    --duration-seconds 60 \
    --out results_2rps.jsonl

# Test multiple arrival rates in one run
python probe_arrival_rate.py \
    --dataset /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --tokenizer hf-internal-testing/llama-tokenizer \
    --total-loras 128 \
    --prompt-regime short \
    --arrival-rates "0.5,1,2,4,8,12,16" \
    --duration-seconds 60 \
    --out results.jsonl
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--endpoint`, `-e` | Server URL | `http://localhost:8000` |
| `--dataset` | Path to ShareGPT JSON file | **Required** |
| `--tokenizer` | Tokenizer name/path | **Required** |
| `--total-loras` | Total LoRA adapters | `128` |
| `--prompt-regime` | `short` (256-1024 tokens) or `long` (2048-4096) | `short` |
| `--arrival-rate`, `-r` | Requests per second | `1.0` |
| `--arrival-rates` | Comma-separated rates for batch testing | - |
| `--duration-seconds`, `-d` | Test duration | `60` |
| `--timeout`, `-t` | Request timeout | `120` |
| `--out`, `-o` | Output file (JSONL or CSV) | - |
| `--max-tokens` | Tokens to generate per request | `16` |
| `--seed` | Random seed for reproducibility | `42` |

### Recommended Testing Workflow

```bash
# Set common variables
DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
TOKENIZER=hf-internal-testing/llama-tokenizer

# 1. Start server with minimal HBM budget
python run_server_single_engine.py \
    --model NousResearch/Llama-2-7b-hf \
    --tokenizer $TOKENIZER \
    --total-loras 128 --gpu-loras 2 \
    --port 8000 --use-dummy-weights

# 2. (In another terminal) Probe arrival rates for SHORT prompts
python probe_arrival_rate.py \
    --dataset $DATASET \
    --tokenizer $TOKENIZER \
    --total-loras 128 \
    --prompt-regime short \
    --arrival-rates "0.5,1,2,4,8" \
    --duration-seconds 60 \
    --out probe_short_gpu2.jsonl

# 3. Restart server with full HBM budget
python run_server_single_engine.py \
    --model NousResearch/Llama-2-7b-hf \
    --tokenizer $TOKENIZER \
    --total-loras 128 --gpu-loras 128 \
    --port 8000 --use-dummy-weights

# 4. Probe same arrival rates
python probe_arrival_rate.py \
    --dataset $DATASET \
    --tokenizer $TOKENIZER \
    --total-loras 128 \
    --prompt-regime short \
    --arrival-rates "0.5,1,2,4,8" \
    --duration-seconds 60 \
    --out probe_short_gpu128.jsonl

# 5. Repeat steps 1-4 with --prompt-regime long
```

### Interpreting Results

The script outputs a comparative summary table:

```
================================================================================
COMPARATIVE SUMMARY (short prompts)
================================================================================
    Rate | Success% |   Tokens |   TTFT Med |   TTFT P99 |    Lat Med |       Status
---------+----------+----------+------------+------------+------------+-------------
     0.5 |   100.0% |      512 |     45.2ms |    123.4ms |     89.1ms |           OK
     1.0 |   100.0% |      508 |     52.3ms |    156.7ms |     98.2ms |           OK
     2.0 |   100.0% |      521 |     78.4ms |    234.5ms |    123.4ms |           OK
     4.0 |    99.2% |      515 |    156.7ms |    890.2ms |    234.5ms |           OK
     8.0 |    97.5% |      510 |    567.8ms |   3456.7ms |    678.9ms |     TOO HIGH
================================================================================
```

- **Tokens**: Average prompt token count (confirms you're testing the right regime)
- **Status values**:
  - `OK`: Acceptable for experiments
  - `UNSTABLE`: p99/median > 5× (high variance)
  - `SLOW`: p99 TTFT > 10 seconds
  - `TOO HIGH`: Error rate > 1%

**Choose the highest rate marked "OK"** as your experiment arrival rate.

### Output Format

JSONL output contains:
- **metadata**: Test configuration
- **summary**: Aggregate statistics (median, p90, p99, error rates)

---

## TTFT/Latency Measurement (Step 3)

After selecting an arrival rate, use `ttft_latency_loras.py` to measure TTFT with different GPU LoRA budgets.

### Quick Start

```bash
# Set common variables
DATASET=/path/to/ShareGPT_V3_unfiltered_cleaned_split.json
TOKENIZER=hf-internal-testing/llama-tokenizer

# Run measurement for a specific gpu_loras setting
python ttft_latency_loras.py \
    --dataset $DATASET \
    --tokenizer $TOKENIZER \
    --total-loras 128 \
    --gpu-loras 32 \
    --prompt-regime short \
    --arrival-rate 2.0 \
    --num-requests 200 \
    --out ttft_results.csv
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--endpoint`, `-e` | Server URL | `http://localhost:8000` |
| `--dataset` | Path to ShareGPT JSON file | **Required** |
| `--tokenizer` | Tokenizer name/path | **Required** |
| `--total-loras` | Total LoRA adapters (CPU-side) | **Required** |
| `--gpu-loras` | LoRA adapters in GPU HBM | **Required** |
| `--prompt-regime` | `short` (256-1024) or `long` (2048-4096) | `short` |
| `--arrival-rate`, `-r` | Requests per second | `2.0` |
| `--num-requests`, `-n` | Number of requests | `200` |
| `--timeout`, `-t` | Request timeout | `120` |
| `--out`, `-o` | Output file (appends if exists) | - |
| `--show-swap-stats` | Show LoRA swap stats after run | `false` |

### Full Experiment Workflow

```bash
DATASET=/path/to/ShareGPT.json
TOKENIZER=hf-internal-testing/llama-tokenizer
TOTAL=128
REGIME=short
RATE=2.0

# Sweep through different gpu_loras values
for GPU_LORAS in 128 64 32 16 8 4 2; do
    echo "=== Testing gpu_loras=$GPU_LORAS ==="
    
    # 1. Start server (in background or separate terminal)
    python run_server_single_engine.py \
        --model NousResearch/Llama-2-7b-hf \
        --tokenizer $TOKENIZER \
        --total-loras $TOTAL \
        --gpu-loras $GPU_LORAS \
        --port 8000 \
        --use-dummy-weights &
    
    sleep 30  # Wait for server startup
    
    # 2. Run measurement
    python ttft_latency_loras.py \
        --dataset $DATASET \
        --tokenizer $TOKENIZER \
        --total-loras $TOTAL \
        --gpu-loras $GPU_LORAS \
        --prompt-regime $REGIME \
        --arrival-rate $RATE \
        --num-requests 200 \
        --out ttft_${REGIME}_total${TOTAL}.csv \
        --show-swap-stats
    
    # 3. Kill server
    pkill -f run_server_single_engine
    sleep 5
done
```

### Output Format

Results are appended to a single CSV file:

```csv
run_timestamp,total_loras,gpu_loras,prompt_regime,arrival_rate,num_requests,ttft_median,ttft_p99,...
2024-01-15 10:30:00,128,128,short,2.0,200,0.045,0.123,...
2024-01-15 10:35:00,128,64,short,2.0,200,0.052,0.156,...
2024-01-15 10:40:00,128,32,short,2.0,200,0.078,0.234,...
```

### Expected Results

As `gpu_loras` decreases, you should see:
- **TTFT increases** (more adapters need to be swapped from CPU)
- **Swap stats show more runtime swaps** (use `--show-swap-stats`)

