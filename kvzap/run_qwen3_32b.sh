#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Compare KVzap (MLP) and TriAttention on Qwen/Qwen3-32B, run from the repository root:
#   bash kvzap/run_qwen3_32b.sh
# 1. Calibrate the TriAttention stats (no released stats for Qwen3-32B)
# 2. Speed (context prefill, TTFT, decoding, throughput, memory) and quick quality check, layers split across GPUs.
#    TriAttention uses the compression ratio reached by KVzap.
# 3. Quality on LooGLE and RULER with the evaluation CLI (device_map="auto" spreads the model over the GPUs)
# Qwen3-32B in bf16 needs ~66 GB: NUM_GPUS=2 with A100 80GB, NUM_GPUS=4 with A100 40GB.

set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-32B}
NUM_GPUS=${NUM_GPUS:-2}
THRESHOLD=${THRESHOLD:--4}
STATS=${STATS:-calibration/qwen3_32b.pt}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-16000}
N_QUESTIONS=${N_QUESTIONS:-8}
FRACTION=${FRACTION:-1.0}  # fraction of the evaluation datasets, e.g. 0.1 for a quick run

if [ ! -f "$STATS" ]; then
    python -m kvzap.calibrate_triattention --model_name "$MODEL" --num_gpus "$NUM_GPUS" --output_file "$STATS"
fi

python -m kvzap.benchmark_sharded --model_name "$MODEL" --num_gpus "$NUM_GPUS" --methods none,kvzap,triattention \
    --kvzap_model_type mlp --threshold "$THRESHOLD" --triattention_stats "$STATS" --match_kvzap_compression True \
    --context_length "$CONTEXT_LENGTH" --n_questions "$N_QUESTIONS"

# Use the KVzap compression ratio measured by the benchmark for TriAttention
RATIO=$(python -c "
import json
results = json.load(open('results/benchmark_sharded__${MODEL//\//--}.json'))['results']
print(next(r['compression_ratio_context'] for r in results if r['method'] == 'kvzap'))")
echo "TriAttention compression ratio for the evaluation: $RATIO"

STATS_PATH=$(realpath "$STATS")
cd evaluation
for DATASET in "loogle shortdep_qa" "ruler 4096"; do
    set -- $DATASET
    python evaluate.py --model "$MODEL" --dataset "$1" --data_dir "$2" --fraction "$FRACTION" --press_name no_press
    python evaluate.py --model "$MODEL" --dataset "$1" --data_dir "$2" --fraction "$FRACTION" \
        --press_name kvzap_mlp --threshold "$THRESHOLD"
    python evaluate.py --model "$MODEL" --dataset "$1" --data_dir "$2" --fraction "$FRACTION" \
        --press_name triattention --compression_ratio "$RATIO" --triattention_stats "$STATS_PATH"
done
