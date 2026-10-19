# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark KV cache compression methods on a model whose decoder layers are split across several GPUs.

For a single long context and several questions about it, every method is measured with the context cache
either reused or recomputed:
    method {none, kvzap, triattention} x context cache {reuse, recompute}

- none: no compression
- kvzap: DMSPress(KVzapPress), evicts the KV pairs whose KVzap score is below a threshold
- triattention: TriAttentionPress, keeps the (1 - compression_ratio) fraction of the context KV pairs with the
  highest TriAttention scores (https://github.com/WeianMao/triattention, needs its calibration stats)

- reuse: the context is prefilled once and its KV cache is shared by all questions. After each answer the
  question and answer tokens are removed from the cache (as in KVPressTextGenerationPipeline).
- recompute: the context is prefilled again from scratch for every question.

Reported metrics:
- context prefill time (and tokens/s)
- TTFT per question: time from submitting the question to its first generated token. With reuse it only covers
  the question forward pass; with recompute it also includes the context prefill.
- decode: mean per-token latency and tokens/s for the tokens after the first one
- end-to-end throughput: generated tokens / total wall time for all questions
- compression ratio and peak memory per GPU

Note that DMSPress does not physically remove KV pairs: evicted keys are masked in the attention
(see kvpress/attention_patch.py), so the KV cache keeps its full size and compression adds a small overhead.
TriAttentionPress removes the pruned KV pairs from the cache, which reduces its size and the attention cost.

Usage (from the repository root, on 2 x A100):
    git clone https://github.com/WeianMao/triattention.git ../triattention
    python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-8B --num_gpus 2 --threshold -4 \
        --triattention_stats ../triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt
"""

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DynamicCache, pipeline

from kvpress import BasePress, DMSPress, KVzapPress, TriAttentionPress
from kvzap.sharding import describe_device_map, load_sharded_model, synchronize_all

logger = logging.getLogger(__name__)


@dataclass
class QuestionStats:
    ttft_s: float
    question_prefill_s: float
    context_prefill_s: float
    decode_token_latencies_s: list[float] = field(repr=False)
    n_generated_tokens: int = 0


@dataclass
class RunStats:
    method: str
    mode: str
    context_tokens: int
    n_questions: int
    total_wall_s: float
    total_generated_tokens: int
    context_prefill_s_mean: float
    context_prefill_tok_per_s: float
    ttft_s_mean: float
    ttft_s_first: float
    decode_ms_per_token: float
    decode_tok_per_s: float
    e2e_tok_per_s: float
    compression_ratio_context: float
    kv_cache_gb: float
    peak_memory_gb: dict[str, float]
    answers: list[str] = field(repr=False)


def load_context_and_questions(context_file: Optional[str], n_questions: int) -> tuple[str, list[str]]:
    """Return a long context and `n_questions` questions about it (from a file or from LooGLE shortdep_qa)."""
    if context_file is not None:
        context = Path(context_file).read_text()
        questions = [f"Summarize part {i + 1} of the document above in one sentence." for i in range(n_questions)]
        return context, questions

    df = load_dataset("simonjegou/loogle", data_dir="shortdep_qa", split="test").to_pandas()
    counts = df.groupby("context")["question"].count()
    candidates = counts[counts >= n_questions]
    assert len(candidates) > 0, f"No LooGLE context with at least {n_questions} questions"
    context = str(candidates.index[0])
    questions = df[df["context"] == context]["question"].tolist()[:n_questions]
    return context, questions


def kv_cache_gb(cache: DynamicCache) -> float:
    return sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers) / 1e9


def context_compression_ratio(press: Optional[BasePress], cache: DynamicCache, context_length: int) -> float:
    """Fraction of the context KV pairs evicted: masked for DMSPress, removed from the cache for other presses."""
    if isinstance(press, DMSPress):
        return press.compression_ratio
    return 1 - sum(cache.get_seq_length(i) for i in range(len(cache))) / (len(cache) * context_length)


@torch.inference_mode()
def prefill_context(model, context_ids: torch.Tensor, press: Optional[BasePress]) -> tuple[DynamicCache, float]:
    cache = DynamicCache()
    synchronize_all()
    start = time.perf_counter()
    if press is not None:
        with press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
    else:
        model.model(input_ids=context_ids, past_key_values=cache)
    synchronize_all()
    return cache, time.perf_counter() - start


@torch.inference_mode()
def answer_question(
    model,
    tokenizer,
    cache: DynamicCache,
    question_ids: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
    ignore_eos: bool,
) -> tuple[str, float, list[float]]:
    """
    Greedy decoding as in KVPressTextGenerationPipeline.generate_answer, timing the question prefill
    (which produces the first token) and every following decoding step.
    """
    position_ids = torch.arange(context_length, context_length + question_ids.shape[1], device=model.device)[None]

    synchronize_all()
    start = time.perf_counter()
    outputs = model(input_ids=question_ids, past_key_values=cache, position_ids=position_ids, logits_to_keep=1)
    generated_ids = [outputs.logits[0, -1].argmax()]
    synchronize_all()
    question_prefill_s = time.perf_counter() - start

    eos_token_ids = model.generation_config.eos_token_id
    eos_token_ids = eos_token_ids if isinstance(eos_token_ids, list) else [eos_token_ids]
    position_ids = position_ids[:, -1:] + 1
    decode_latencies = []
    for i in range(max_new_tokens - 1):
        if not ignore_eos and generated_ids[-1].item() in eos_token_ids:
            break
        start = time.perf_counter()
        outputs = model(
            input_ids=generated_ids[-1].to(model.device)[None, None],
            past_key_values=cache,
            position_ids=position_ids + i,
        )
        generated_ids.append(outputs.logits[0, -1].argmax())
        synchronize_all()
        decode_latencies.append(time.perf_counter() - start)

    answer = tokenizer.decode(torch.stack(generated_ids), skip_special_tokens=True)
    return answer, question_prefill_s, decode_latencies


@torch.inference_mode()
def run(
    pipe,
    method: str,
    context_ids: torch.Tensor,
    questions_ids: list[torch.Tensor],
    press: Optional[BasePress],
    reuse: bool,
    max_new_tokens: int,
    ignore_eos: bool,
) -> RunStats:
    model, tokenizer = pipe.model, pipe.tokenizer
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)
    decoding = isinstance(press, DMSPress) and press.decoding
    context_length = context_ids.shape[1]

    question_stats, answers, context_prefill_times = [], [], []
    compression_ratio_context, cache_gb = 0.0, 0.0
    cache, dms_state = None, None

    synchronize_all()
    run_start = time.perf_counter()
    for question_idx, question_ids in enumerate(questions_ids):
        context_prefill_s = 0.0
        if not reuse or question_idx == 0:
            cache, context_prefill_s = prefill_context(model, context_ids, press)
            context_prefill_times.append(context_prefill_s)
            # With decoding=True, DMSPress also evicts question and answer tokens: save its state after the
            # context prefill to restore it before the next question, as KVPressTextGenerationPipeline does
            dms_state = pipe._save_dms_state(press, cache) if decoding else None
            cache_gb = kv_cache_gb(cache)
            compression_ratio_context = context_compression_ratio(press, cache, context_length)
        elif dms_state is not None:
            pipe._restore_dms_state(press, dms_state, cache)
        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]

        if decoding:
            with press(model):
                answer, question_prefill_s, decode_latencies = answer_question(
                    model, tokenizer, cache, question_ids, context_length, max_new_tokens, ignore_eos
                )
        else:
            answer, question_prefill_s, decode_latencies = answer_question(
                model, tokenizer, cache, question_ids, context_length, max_new_tokens, ignore_eos
            )

        question_stats.append(
            QuestionStats(
                ttft_s=context_prefill_s + question_prefill_s,
                question_prefill_s=question_prefill_s,
                context_prefill_s=context_prefill_s,
                decode_token_latencies_s=decode_latencies,
                n_generated_tokens=len(decode_latencies) + 1,
            )
        )
        answers.append(answer)

        # Remove the question and answer from the context cache before the next question
        if reuse:
            for layer, seq_len in zip(cache.layers, cache_seq_lengths):
                layer.keys = layer.keys[:, :, :seq_len]
                layer.values = layer.values[:, :, :seq_len]
    synchronize_all()
    total_wall_s = time.perf_counter() - run_start

    all_decode = [t for s in question_stats for t in s.decode_token_latencies_s]
    total_tokens = sum(s.n_generated_tokens for s in question_stats)
    return RunStats(
        method=method,
        mode="reuse" if reuse else "recompute",
        context_tokens=context_length,
        n_questions=len(questions_ids),
        total_wall_s=total_wall_s,
        total_generated_tokens=total_tokens,
        context_prefill_s_mean=mean(context_prefill_times),
        context_prefill_tok_per_s=context_length / mean(context_prefill_times),
        ttft_s_mean=mean(s.ttft_s for s in question_stats),
        ttft_s_first=question_stats[0].ttft_s,
        decode_ms_per_token=1e3 * mean(all_decode) if all_decode else float("nan"),
        decode_tok_per_s=len(all_decode) / sum(all_decode) if all_decode else float("nan"),
        e2e_tok_per_s=total_tokens / total_wall_s,
        compression_ratio_context=compression_ratio_context,
        kv_cache_gb=cache_gb,
        peak_memory_gb={
            f"cuda:{i}": torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())
        },
        answers=answers,
    )


def print_table(results: list[RunStats]):
    header = (
        f"{'method':<14}{'mode':<11}{'ctx prefill s':>14}{'TTFT mean s':>13}{'TTFT q1 s':>11}"
        f"{'decode ms/tok':>15}{'decode tok/s':>14}{'e2e tok/s':>11}{'total s':>9}{'comp. ratio':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.method:<14}{r.mode:<11}{r.context_prefill_s_mean:>14.3f}{r.ttft_s_mean:>13.3f}"
            f"{r.ttft_s_first:>11.3f}{r.decode_ms_per_token:>15.2f}{r.decode_tok_per_s:>14.1f}"
            f"{r.e2e_tok_per_s:>11.1f}{r.total_wall_s:>9.2f}{r.compression_ratio_context:>13.2%}"
        )
    for r in results:
        memory = ", ".join(f"{device}: {gb:.1f} GB" for device, gb in r.peak_memory_gb.items())
        print(f"peak memory [{r.method}/{r.mode}]: {memory} (KV cache {r.kv_cache_gb:.2f} GB)")


def benchmark(
    model_name: str = "Qwen/Qwen3-8B",
    methods: str | tuple[str, ...] = ("none", "kvzap", "triattention"),
    kvzap_model_type: str = "mlp",
    threshold: float = -4.0,
    decoding: bool = False,
    triattention_stats: Optional[str] = None,
    triattention_compression_ratio: float = 0.5,
    num_gpus: int = 2,
    context_length: int = 16000,
    n_questions: int = 4,
    max_new_tokens: int = 256,
    ignore_eos: bool = True,
    context_file: Optional[str] = None,
    output_file: Optional[str] = None,
):
    """Benchmark compression methods and context cache reuse vs recompute on a layer-sharded model.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model name, by default "Qwen/Qwen3-8B" (needs a matching nvidia/KVzap-* model)
    methods : str or tuple of str, optional
        Methods to compare among "none", "kvzap" and "triattention", by default all of them
    kvzap_model_type : str, optional
        KVzap surrogate model type, "mlp" or "linear", by default "mlp"
    threshold : float, optional
        DMSPress threshold on the KVzap scores, by default -4.0
    decoding : bool, optional
        Also compress the question and generated tokens (DMSPress.decoding), by default False
    triattention_stats : str, optional
        Path to the TriAttention calibration stats for model_name, required for "triattention"
    triattention_compression_ratio : float, optional
        Fraction of the context KV pairs removed by TriAttention, by default 0.5
    num_gpus : int, optional
        Number of GPUs to split the decoder layers across, by default 2
    context_length : int, optional
        Maximum number of context tokens (the context is truncated), by default 16000
    n_questions : int, optional
        Number of questions asked about the context, by default 4
    max_new_tokens : int, optional
        Number of tokens generated per question, by default 256
    ignore_eos : bool, optional
        Keep generating after EOS so that every run decodes the same number of tokens, by default True
    context_file : str, optional
        Text file used as context instead of a LooGLE document, by default None
    output_file : str, optional
        Where to save the JSON results, by default results/benchmark_sharded__<model>.json
    """
    methods = methods.split(",") if isinstance(methods, str) else list(methods)
    presses: dict[str, Optional[BasePress]] = {}
    for method in methods:
        if method == "none":
            presses[method] = None
        elif method == "kvzap":
            presses[method] = DMSPress(KVzapPress(model_type=kvzap_model_type), threshold=threshold, decoding=decoding)
        elif method == "triattention":
            assert triattention_stats is not None, "--triattention_stats is required for the triattention method"
            presses[method] = TriAttentionPress(
                compression_ratio=triattention_compression_ratio, stats_path=triattention_stats
            )
        else:
            raise ValueError(f"Unknown method {method}, expected none, kvzap or triattention")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_sharded_model(model_name, num_gpus=num_gpus)
    print(f"Sharded {model_name}: {describe_device_map(model)}")

    # Reuse the pipeline preprocessing (chat template, context truncation) so that the inputs match evaluation
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
    context, questions = load_context_and_questions(context_file, n_questions)
    inputs = pipe.preprocess(context, questions, answer_prefix="", max_context_length=context_length)
    context_ids = inputs["context_ids"].to(model.device)
    questions_ids = [q.to(model.device) for q in inputs["questions_ids"]]
    print(f"Context: {context_ids.shape[1]} tokens, {len(questions_ids)} questions, {max_new_tokens} new tokens each")

    # Warmup: loads the KVzap model / TriAttention stats and the CUDA kernels on all GPUs
    for method, press in presses.items():
        run(pipe, method, context_ids[:, :1024], questions_ids[:1], press, True, 8, ignore_eos)

    results = []
    for method, press in presses.items():
        for reuse in [True, False]:
            results.append(run(pipe, method, context_ids, questions_ids, press, reuse, max_new_tokens, ignore_eos))
            r = results[-1]
            print(f"done: method={r.method} mode={r.mode} total={r.total_wall_s:.2f}s")

    print()
    print_table(results)

    output_path = Path(output_file or f"results/benchmark_sharded__{model_name.replace('/', '--')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "methods": methods,
        "kvzap_model_type": kvzap_model_type,
        "threshold": threshold,
        "decoding": decoding,
        "triattention_stats": triattention_stats,
        "triattention_compression_ratio": triattention_compression_ratio,
        "num_gpus": num_gpus,
        "device_map": describe_device_map(model),
        "max_new_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
    }
    with open(output_path, "w") as f:
        json.dump({"config": config, "results": [asdict(r) for r in results]}, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
