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

`benchmark_sharded.py` splits the decoder layers across `--num_gpus` GPUs and measures context prefill time, TTFT, decoding speed and end-to-end throughput, with the context KV cache either reused across questions or recomputed for every question. It compares no compression, KVzap (`DMSPress(KVzapPress)`) and [TriAttention](https://arxiv.org/abs/2604.04921) (`TriAttentionPress`), which needs the calibration statistics from the [official repository](https://github.com/WeianMao/triattention):

```bash
git clone https://github.com/WeianMao/triattention.git ../triattention
python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-8B --num_gpus 2 --context_length 16000 --n_questions 4 \
    --threshold -4 --triattention_compression_ratio 0.5 \
    --triattention_stats ../triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt
```

Use `--methods none,kvzap` to skip TriAttention, and `--decoding True` to also compress the question and generated tokens with KVzap. Since `DMSPress` masks evicted keys rather than removing them from the cache, KVzap does not reduce the KV cache size or speed up decoding, whereas `TriAttentionPress` removes the pruned KV pairs.

The table also reports a quick quality check: answers are scored with the token F1 against the LooGLE reference answers (`F1 ref`), and against the answers without compression (`F1 none`, `EM none`). Add `--match_kvzap_compression True` to run TriAttention at the context compression ratio reached by KVzap, so that both methods are compared at the same compression.

### Qwen3-32B

KVzap provides `nvidia/KVzap-mlp-Qwen3-32B`, but TriAttention has no released calibration statistics for Qwen3-32B. `calibrate_triattention.py` computes them (single forward pass on ~32k tokens of the Nemotron pretraining sample, or `--text_file`), with the queries captured after the Qwen3 `q_norm`, right before RoPE:

```bash
python -m kvzap.calibrate_triattention --model_name Qwen/Qwen3-32B --num_gpus 2 --output_file calibration/qwen3_32b.pt
python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-32B --num_gpus 2 --n_questions 8 \
    --threshold -4 --triattention_stats calibration/qwen3_32b.pt --match_kvzap_compression True
```

For quality on full datasets, the evaluation CLI supports TriAttention with `--press_name triattention --triattention_stats <stats>`:

```bash
cd evaluation
python evaluate.py --model Qwen/Qwen3-32B --dataset loogle --data_dir shortdep_qa --press_name kvzap_mlp --threshold -4
python evaluate.py --model Qwen/Qwen3-32B --dataset loogle --data_dir shortdep_qa --press_name triattention \
    --compression_ratio 0.5 --triattention_stats ../calibration/qwen3_32b.pt
```

`run_qwen3_32b.sh` runs all these steps (calibration, speed benchmark, and LooGLE / RULER evaluation of no compression, KVzap MLP and TriAttention at the KVzap compression ratio). Qwen3-32B in bf16 needs ~66 GB of weights: use `NUM_GPUS=2` with 80 GB GPUs and `NUM_GPUS=4` with 40 GB GPUs.
