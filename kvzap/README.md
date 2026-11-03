# KVzap

[![KVzap collection](https://img.shields.io/badge/🤗%20Hugging%20Face-Collection-orange)](https://huggingface.co/collections/nvidia/kvzap) 
[![arXiv](https://img.shields.io/badge/arXiv-2601.07891-b31b1b.svg)](https://arxiv.org/abs/2601.07891)

[KVzap](https://arxiv.org/abs/2601.07891) is a fast approximation of [KVzip](https://arxiv.org/abs/2505.23416) that works in both prefilling and decoding. It applies a lightweight surrogate model to the hidden states to predict importance scores, and removes the KV pairs with a score below a given threshold, following the Dynamic Memory Sparsification ([DMS](https://arxiv.org/abs/2506.05345)) inference strategy.

## Usage

KVzap is designed to be used by combining the `KVzapPress` and the `DMSPress` from kvpress:

```python
import requests
from transformers import pipeline
from kvpress import KVzapPress, DMSPress

model = "Qwen/Qwen3-8B"
pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", dtype="auto")
press = DMSPress(KVzapPress(model_type="mlp"), threshold=-4)

# Prefilling compression only, thinking disabled
press.decoding = False
context = requests.get("https://arxiv.org/abs/2601.07891").text
question = "\n What is this article about in 2 sentences ?"
answer = pipe(context, question=question, press=press)["answer"]
print(f"Compression ratio: {press.compression_ratio:.2%}\nAnswer: {answer}")

# Prefilling and decoding compression, thinking enabled
press.decoding = True
prompt = "What is the best hardware to run LLMs and why ?"
answer = pipe(prompt, press=press, enable_thinking=True, max_new_tokens=2000)["answer"]
print(f"Compression ratio: {press.compression_ratio:.2%}\nAnswer: {answer}")
```

The `KVzapPress` inherits from the `ScorerPress` class and only predicts the scores for every KV pair. The `DMSPress` then prunes the KV pairs with a score below a given threshold, rather than using a fixed compression ratio.

Supported base models are provided in the [KVzap collection](https://huggingface.co/collections/nvidia/kvzap) but can easily be extended to any other model following the instructions in the [training section](#training).

## Training

Training uses the [Nemotron-Pretraining-Dataset-sample](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Dataset-sample) to extract KVzip+ scores and train surrogate models. 

To reproduce the training or train your own model, use the following command:

```bash
pip install skorch scikit-learn
python train.py --model_name <model_name> --output_dir <output_dir>
```

Run `python train.py --help` for all options.

## Evaluation

Evaluation can be reproduced by using the [kvpress evaluation CLI](../evaluation). 

We provide a specific script to evaluate KVzap on the AIME25 benchmark using `model.generate` directly to enable sampling-based decoding rather than greedy decoding:

```bash
python evaluate_aime.py <model_type> --threshold <threshold> --model_name <base_model_name>
```

where `<model_type>` is the type of KVzap model to use ("mlp", "linear" or "no_press") and `<base_model_name>` the name of the base model to use (e.g. "Qwen/Qwen3-8B"). Add `--num_gpus 2` to split the decoder layers of the base model across 2 GPUs.

## Multi-GPU benchmark

`benchmark_sharded.py` splits the decoder layers across `--num_gpus` GPUs and measures context prefill time, TTFT, decoding speed and end-to-end throughput, with the context KV cache either reused across questions or recomputed for every question. It compares no compression, physical KVzap pruning (`KVzapPress` with an MLP surrogate) and [TriAttention](https://arxiv.org/abs/2604.04921) (`TriAttentionPress`), which needs the calibration statistics from the [official repository](https://github.com/WeianMao/triattention):

```bash
git clone https://github.com/WeianMao/triattention.git ../triattention
python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-8B --num_gpus 2 --context_length 16000 --n_questions 4 \
    --kvzap_compression_ratio 0.5 --triattention_compression_ratio 0.5 \
    --triattention_stats ../triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt
```

Use `--methods none,kvzap` to skip TriAttention. In this benchmark, KVzap always uses its MLP surrogate to rank context KV entries and physically keeps the top-scoring entries per layer and KV head. Both methods reduce the actual cache tensors after prefill; question and generated tokens are not pruned. A compression ratio of `0.5` removes approximately half of the context KV entries, subject to integer rounding. There is no protected recent-token window in this fixed-budget policy.

Use `--compression_ratios 0.25,0.5,0.75` to sweep the same physical removal budgets for both KVzap and TriAttention. Otherwise, `--kvzap_compression_ratio` and `--triattention_compression_ratio` select their individual budgets. The former benchmark flags `--threshold`, `--thresholds`, `--decoding`, and `--kvzap_model_type` have been removed: threshold-based DMS masking and the linear surrogate are not used by this benchmark. The standalone library's `DMSPress` API and the other evaluation scripts are unchanged.

Quality scoring is enabled by default. Use `--quality True --divergence_tokens 32 --ignore_eos False` to score reference answers and compare dense/compressed next-token distributions at up to 32 dense continuation positions. LooGLE references are loaded automatically. JSON reports record achieved physical compression, actual KV bytes, timing, peak allocations, per-sample quality, and per-token KL/JS/NLL/agreement; plots are saved beside the JSON. Fixed-budget reports use schema version 3 and label KVzap as `physical_topk`; reports including the adaptive policy use schema version 4. Historical masking reports are not modified and should not be combined with these runs as the same policy.

The sharded loader checks cross-GPU tensor copies before loading the model. If direct copies corrupt values, it warns and stages transfers through CPU memory. Use `--cpu_staging True` to force this fallback for intermittent peer-copy failures. This avoids corrupted activations and inconsistent cache lengths, but adds transfer overhead to benchmark timings. The JSON configuration records whether CPU staging was used.

The table also reports reference token F1 (`F1 ref`) and agreement with the dense answers (`F1 none`, `EM none`), paired within the same context, length, repetition, and cache mode. Answers stop at the first EOS even when `--ignore_eos True` keeps generation running for timing. Add `--match_kvzap_compression True` to run each TriAttention sweep point at the physical context compression reached by the corresponding KVzap point; this overrides the separate TriAttention budget. Actual compression is recorded per run because integer rounding can vary with context length.

### Adaptive KVzap MLP

Add `adaptive_kvzap` to `--methods` to evaluate the integrated policy from the sibling `../adaptive_kvcomp_poc/adaptive_kvpress_poc/policies/` checkout. It requires the real-score `CompressionPolicy.decide_scores` API in that checkout. Use `--adaptive_policy_root /path/to/adaptive_kvcomp_poc` for another location; only adaptive runs import this dependency. The existing fixed-budget `kvzap` method is unchanged.

The adaptive path uses real pretrained KVzap MLP scores, not the PoC's synthetic score generator. It averages scores across KV heads within each layer, applies the integrated threshold/window/reuse policy, and physically gathers the retained positions in chronological order. All heads in that layer share the exact same mask, avoiding padding variable per-head lengths. Thresholds and masks can differ across layers. Adaptive runs default to FlashAttention 2 and reject other attention backends because SDPA's shared causal mask does not support these variable per-layer cache lengths. Question and generated tokens are not pruned, and no decoding-time eviction is added.

The surrogate selector runs with the requested compression factor, task, and latency budget, but its recommendation is advisory: this experiment deliberately forces the actual scorer to MLP, including when the selector recommends a heuristic, linear model, or ensemble. Reports record both recommended and actual tiers. This is not an evaluation of dynamic scorer switching or CriticalKV refinement.

From the repository root, a bounded-context comparison on two A100 40GB GPUs is:

```bash
.venv/bin/python -m kvzap.benchmark_sharded \
    --model_name Qwen/Qwen3-32B --num_gpus 2 --cpu_staging True \
    --attn_implementation flash_attention_2 --local_files_only True \
    --methods none,kvzap,adaptive_kvzap --kvzap_compression_ratio 0.5 \
    --adaptive_target_compression 2.0 --context_length 4096 \
    --n_contexts 1 --n_questions 2 --max_new_tokens 32 \
    --quality True --divergence_tokens 4 --ignore_eos False \
    --output_file results/qwen3_32b_adaptive.json
```

`--adaptive_target_compression 2.0` means a nominal 2x size reduction, not a 2.0 removal fraction or a guaranteed budget. Without that option, the factor is derived from the KVzap ratio as `1 / (1 - ratio)`. Alternatively, `--compression_ratios 0.25,0.5,0.75` compares fixed KVzap removal budgets with adaptive targets of approximately 1.33x, 2x, and 4x; omit `adaptive_target_compression` when sweeping. Compare achieved physical compression, not nominal targets. The PoC threshold constants were calibrated on synthetic distributions and are not quality guarantees for Qwen3-32B.

`--adaptive_threshold adaptive|fixed|percentile` chooses the existing threshold class. `--adaptive_tau` controls fixed mode; `--adaptive_entropy_sensitivity` controls adaptive mode. Adaptive targets above 4x are rejected because the PoC lookup table is only calibrated through 4x and its overflow fallback is not monotonic. Percentile mode retains the PoC's approximate quantile interpolation rather than exact top-k. Other controls are `--adaptive_task_type`, `--adaptive_latency_budget_ms`, `--adaptive_base_window`, `--adaptive_head_tokens`, `--adaptive_max_window`, `--adaptive_boundary_extension`, `--adaptive_scale_window`, and `--adaptive_score_aware_boundary`. Statistics, including 50-bin histogram entropy in nats, are recomputed from reuse-adjusted real scores.

Reuse-aware bonuses are enabled by default. Causal prefix hashes are computed from the actual preprocessed context tokens in 512-token blocks. Frequency counts planned question requests separately for each context-length sweep, excluding timing repetitions, methods, and reuse/recompute modes. Thus the same workload metadata is used for both modes. Set `--adaptive_reuse_aware False` for an ablation or change `--adaptive_reuse_bonus_scale`. These are workload-informed bonuses, not an online prefix-cache implementation or a measured cache-hit improvement.

Schema-4 reports include policy source paths/hashes, per-prefill/per-layer decisions, thresholds, score statistics, actual/recommended tiers, window sizes, protected-token counts, and achieved compression. Summary metrics include threshold/window averages and prefix-block survival at the PoC's 80% criterion; this criterion does not mean an intact, reusable KV block. Policy CPU transfers and decision time are included in timed prefills. A one-context run is a smoke check, not a quality study; increase contexts and repeats for evaluation. Both GPUs must have enough free memory for the roughly 66 GB BF16 base model plus working memory. `local_files_only` covers the base model/tokenizer, not dataset access or the surrogate's separate Hugging Face loader.

### Qwen3-32B

KVzap provides `nvidia/KVzap-mlp-Qwen3-32B`, but TriAttention has no released calibration statistics for Qwen3-32B. `calibrate_triattention.py` computes them (single forward pass on ~32k tokens of the Nemotron pretraining sample, or `--text_file`), with the queries captured after the Qwen3 `q_norm`, right before RoPE:

```bash
python -m kvzap.calibrate_triattention --model_name Qwen/Qwen3-32B --num_gpus 2 --output_file calibration/qwen3_32b.pt
python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-32B --num_gpus 2 --n_questions 8 \
    --kvzap_compression_ratio 0.5 --triattention_stats calibration/qwen3_32b.pt --match_kvzap_compression True
```

For quality on full datasets, the evaluation CLI supports TriAttention with `--press_name triattention --triattention_stats <stats>`:

```bash
cd evaluation
python evaluate.py --model Qwen/Qwen3-32B --num_gpus 2 --cpu_staging True \
    --dataset loogle --data_dir shortdep_qa --press_name kvzap_mlp_head \
    --compression_ratio 0.5 --max_context_length 4096 --max_new_tokens 128
python evaluate.py --model Qwen/Qwen3-32B --dataset loogle --data_dir shortdep_qa --press_name triattention \
    --num_gpus 2 --cpu_staging True --compression_ratio 0.5 --max_context_length 4096 \
    --max_new_tokens 128 --triattention_stats ../results/qwen3_32b_stats.pt
```