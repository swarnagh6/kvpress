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

Quality scoring is enabled by default. Use `--quality True --divergence_tokens 32 --ignore_eos False` to score reference answers and compare dense/compressed next-token distributions at up to 32 dense continuation positions. LooGLE references are loaded automatically. JSON reports record achieved physical compression, actual KV bytes, timing, peak allocations, per-sample quality, and per-token KL/JS/NLL/agreement; plots are saved beside the JSON. New reports use schema version 3 and label KVzap as `physical_topk`. Historical masking reports are not modified and should not be combined with these runs as the same policy.

The sharded loader checks cross-GPU tensor copies before loading the model. If direct copies corrupt values, it warns and stages transfers through CPU memory. Use `--cpu_staging True` to force this fallback for intermittent peer-copy failures. This avoids corrupted activations and inconsistent cache lengths, but adds transfer overhead to benchmark timings. The JSON configuration records whether CPU staging was used.
