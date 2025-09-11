#!/bin/bash

export BASE="/fsx/joel_niklaus/projects/finephrase/vllm_benchmark"
export MODEL="Qwen/Qwen3-8B-FP8"
export SYSTEM="GPU"
export TP=1
export DOWNLOAD_DIR=""
export INPUT_LEN=8192
export OUTPUT_LEN=$((4096 + 2048))
export MAX_MODEL_LEN=16384
export MIN_CACHE_HIT_PCT=5  # System prompt is around 500 tokens: 500 / 8000 = 6.25%
export MAX_LATENCY_ALLOWED_MS=100000000000 # A very large number to maximize throughput
export NUM_SEQS_LIST="256"
export NUM_BATCHED_TOKENS_LIST="2048"
export NUM_PROMPTS_MAIN=200
export NUM_PROMPTS_SUB=50
export VLLM_LOGGING_LEVEL="DEBUG"

bash autotune.sh

