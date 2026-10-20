# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark KV cache compression methods on a model whose decoder layers are split across several GPUs.

For a single long context and several questions about it, every method is measured with the context cache
either reused or recomputed:
    method {none, kvzap, triattention} x context cache {reuse, recompute}

- none: no compression
- kvzap: KVzapPress with an MLP surrogate, physically prunes the lowest-scoring context KV pairs
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

Both KVzap and TriAttention physically remove context KV pairs after prefill using fixed-budget top-k selection.
Question and generated tokens are not compressed. Original absolute positions are preserved during generation.

Usage (from the repository root, on 2 x A100):
    git clone https://github.com/WeianMao/triattention.git ../triattention
    python -m kvzap.benchmark_sharded --model_name Qwen/Qwen3-8B --num_gpus 2 --kvzap_compression_ratio 0.5 \
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

from kvpress import BasePress, KVzapPress, TriAttentionPress
from kvzap.benchmark_quality import answer_quality, matched_compression_pairs, summarize_samples, token_divergence
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
    decode_ms_per_token: Optional[float]
    decode_tok_per_s: Optional[float]
    e2e_tok_per_s: float
    compression_ratio_context: float
    kv_cache_gb: float
    peak_memory_gb: dict[str, float]
    answers: list[str] = field(repr=False)
    questions: list[dict] = field(default_factory=list, repr=False)
    setting: str = "default"
    context_id: str = "0"
    repetition: int = 0


def load_benchmark_samples(context_file, samples_file, n_questions, n_contexts):
    if samples_file:
        path = Path(samples_file)
        text = path.read_text()
        records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if path.suffix == ".jsonl"
            else json.loads(text)
        )
        if not isinstance(records, list) or not records:
            raise ValueError("Samples must be a nonempty JSON list or JSONL records")
        samples = []
        for index, record in enumerate(records[:n_contexts]):
            if not isinstance(record.get("context"), str) or not record.get("questions"):
                raise ValueError("Each sample needs a context string and nonempty questions")
            questions, references, depths = [], [], []
            for question in record["questions"][:n_questions]:
                if not isinstance(question, dict) or not isinstance(question.get("question"), str):
                    raise ValueError(
                        "Questions must be objects with a question string and optional answers/needle_depth"
                    )
                answers = question.get("answers", [])
                answers = [answers] if isinstance(answers, str) else answers
                if not isinstance(answers, list) or any(
                    not isinstance(answer, str) or not answer.strip() for answer in answers
                ):
                    raise ValueError("answers must be a string or a list of nonempty reference strings")
                depth = question.get("needle_depth")
                if depth is not None and (not isinstance(depth, (int, float)) or not 0 <= depth <= 1):
                    raise ValueError("needle_depth must be a fraction from 0 to 1, or null")
                questions.append(question["question"])
                references.append(answers)
                depths.append(depth)
            samples.append(
                {
                    "id": str(record.get("id", index)),
                    "context": record["context"],
                    "questions": questions,
                    "references": references,
                    "needle_depths": depths,
                }
            )
        if len({sample["id"] for sample in samples}) != len(samples):
            raise ValueError("Context IDs must be unique")
        return samples
    if context_file is not None:
        context = Path(context_file).read_text()
        questions = [f"Summarize part {i + 1} of the document above in one sentence." for i in range(n_questions)]
        return [
            {
                "id": "text",
                "context": context,
                "questions": questions,
                "references": [[] for _ in questions],
                "needle_depths": [None] * len(questions),
            }
        ]

    df = load_dataset("simonjegou/loogle", data_dir="shortdep_qa", split="test").to_pandas()
    counts = df.groupby("context")["question"].count()
    candidates = counts[counts >= n_questions]
    if len(candidates) < n_contexts:
        raise ValueError(f"Need {n_contexts} LooGLE contexts with at least {n_questions} questions")
    samples = []
    for index, context in enumerate(candidates.index[:n_contexts]):
        rows = df[df["context"] == context].iloc[:n_questions]
        references = [[answer] if isinstance(answer, str) else list(answer) for answer in rows["answer"]]
        samples.append(
            {
                "id": f"loogle-{index}",
                "context": str(context),
                "questions": rows["question"].tolist(),
                "references": references,
                "needle_depths": [None] * len(rows),
            }
        )
    return samples


def load_context_and_questions(context_file: Optional[str], n_questions: int) -> tuple[str, list[str]]:
    sample = load_benchmark_samples(context_file, None, n_questions, 1)[0]
    return sample["context"], sample["questions"]


def parse_sweep(value, default, cast=float):
    if value is None:
        return [default]
    values = value.split(",") if isinstance(value, str) else value
    values = list(dict.fromkeys(cast(item) for item in (values if isinstance(values, (list, tuple)) else [values])))
    if not values or any(not torch.isfinite(torch.tensor(item)) for item in values):
        raise ValueError("Sweep values must be a nonempty list of finite numbers")
    return values


def kv_cache_gb(cache: DynamicCache) -> float:
    return sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers) / 1e9


def context_compression_ratio(cache: DynamicCache, context_length: int) -> float:
    """Fraction of context positions physically removed, averaged across layers."""
    return 1 - sum(cache.get_seq_length(i) for i in range(len(cache))) / (len(cache) * context_length)


def cache_token_metrics(cache, context_length: int) -> dict:
    layers = []
    for index, layer in enumerate(cache.layers):
        batch, heads, length, _ = layer.keys.shape
        physical = batch * heads * length
        original = batch * heads * context_length
        layers.append(
            {
                "layer": index,
                "original_kv_slots": original,
                "physical_kv_slots": physical,
                "retained_kv_slots": physical,
                "kv_bytes": layer.keys.nbytes + layer.values.nbytes,
            }
        )
    original = sum(layer["original_kv_slots"] for layer in layers)
    physical = sum(layer["physical_kv_slots"] for layer in layers)
    retained = sum(layer["retained_kv_slots"] for layer in layers)
    return {
        "original_kv_slots": original,
        "physical_kv_slots": physical,
        "retained_kv_slots": retained,
        "retained_tokens_per_layer_head": retained / (original / context_length),
        "achieved_compression_ratio": 1 - retained / original,
        "physical_compression_ratio": 1 - physical / original,
        "kv_bytes": sum(layer["kv_bytes"] for layer in layers),
        "layers": layers,
    }


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
    token_sink: Optional[list] = None,
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

    tokens = torch.stack(generated_ids)
    answer = tokenizer.decode(tokens, skip_special_tokens=True)
    if token_sink is not None:
        token_sink.extend(tokens.cpu().tolist())
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
    context_length = context_ids.shape[1]

    question_stats, answers, context_prefill_times, question_records = [], [], [], []
    compression_ratio_context, cache_gb = 0.0, 0.0
    cache = None

    synchronize_all()
    run_start = time.perf_counter()
    for question_idx, question_ids in enumerate(questions_ids):
        context_prefill_s = 0.0
        if not reuse or question_idx == 0:
            cache = None
            cache, context_prefill_s = prefill_context(model, context_ids, press)
            context_prefill_times.append(context_prefill_s)
            cache_gb = kv_cache_gb(cache)
            compression_ratio_context = context_compression_ratio(cache, context_length)
            token_metrics = cache_token_metrics(cache, context_length)
        cache_seq_lengths = [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache))]

        generated_tokens = []
        answer, question_prefill_s, decode_latencies = answer_question(
            model, tokenizer, cache, question_ids, context_length, max_new_tokens, ignore_eos, generated_tokens
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
        question_records.append(
            {
                "question_index": question_idx,
                "generated_token_ids": generated_tokens,
                "ttft_s": context_prefill_s + question_prefill_s,
                "question_prefill_s": question_prefill_s,
                "context_prefill_s": context_prefill_s,
                "cached_context": reuse and question_idx > 0,
                "n_generated_tokens": len(generated_tokens),
                "decode_token_latencies_s": decode_latencies,
                "decode_tok_per_s": len(decode_latencies) / sum(decode_latencies) if decode_latencies else None,
                "decode_ms_per_token": 1e3 * mean(decode_latencies) if decode_latencies else None,
                **token_metrics,
            }
        )

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
        decode_ms_per_token=1e3 * mean(all_decode) if all_decode else None,
        decode_tok_per_s=len(all_decode) / sum(all_decode) if all_decode else None,
        e2e_tok_per_s=total_tokens / total_wall_s,
        compression_ratio_context=compression_ratio_context,
        kv_cache_gb=cache_gb,
        peak_memory_gb={
            f"cuda:{i}": torch.cuda.max_memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())
        },
        answers=answers,
        questions=question_records,
    )


@torch.inference_mode()
def teacher_forced_logits(model, cache, question_ids, continuation: list[int], context_length: int) -> torch.Tensor:
    if not continuation:
        raise ValueError("Divergence needs at least one continuation token")
    positions = torch.arange(context_length, context_length + question_ids.shape[1], device=model.device)[None]
    outputs = model(input_ids=question_ids, past_key_values=cache, position_ids=positions, logits_to_keep=1)
    logits = [outputs.logits[0, -1].float().cpu()]
    next_position = context_length + question_ids.shape[1]
    for offset, token in enumerate(continuation[:-1]):
        outputs = model(
            input_ids=torch.tensor([[token]], device=model.device),
            past_key_values=cache,
            position_ids=torch.tensor([[next_position + offset]], device=model.device),
            logits_to_keep=1,
        )
        logits.append(outputs.logits[0, -1].float().cpu())
    return torch.stack(logits)


def measure_quality(pipe, result, dense_result, context_ids, questions_ids, press, references, divergence_tokens):
    records = []
    for index, (question_ids, prediction) in enumerate(zip(questions_ids, result.answers)):
        metrics = answer_quality(prediction, references[index])
        divergence = {}
        if divergence_tokens:
            continuation = dense_result.questions[index]["generated_token_ids"][:divergence_tokens]
            dense_cache, _ = prefill_context(pipe.model, context_ids, None)
            dense_logits = teacher_forced_logits(
                pipe.model, dense_cache, question_ids, continuation, context_ids.shape[1]
            )
            del dense_cache
            if press is None:
                candidate_logits = dense_logits
            else:
                cache, _ = prefill_context(pipe.model, context_ids, press)
                candidate_logits = teacher_forced_logits(
                    pipe.model, cache, question_ids, continuation, context_ids.shape[1]
                )
                del cache
            divergence = token_divergence(dense_logits, candidate_logits, torch.tensor(continuation))
            metrics.update({name: mean(values) for name, values in divergence.items()})
            metrics["kl_p95_token"] = float(torch.tensor(divergence["kl_dense_to_compressed"]).quantile(0.95))
            metrics["kl_worst_token"] = max(divergence["kl_dense_to_compressed"])
        records.append(
            {
                "question_index": index,
                "references": references[index],
                "prediction": prediction,
                "dense_prediction": dense_result.answers[index],
                "per_token_divergence": divergence,
                "divergence_token_ids": continuation if divergence else [],
                "divergence_prefix": "dense_greedy_continuation",
                "divergence_tokens": len(next(iter(divergence.values()))) if divergence else 0,
                **metrics,
            }
        )
    return records


def print_table(results: list[RunStats]):
    header = (
        f"{'method':<14}{'mode':<11}{'ctx prefill s':>14}{'TTFT mean s':>13}{'TTFT q1 s':>11}"
        f"{'decode ms/tok':>15}{'decode tok/s':>14}{'e2e tok/s':>11}{'total s':>9}{'comp. ratio':>13}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        decode_ms = f"{r.decode_ms_per_token:.2f}" if r.decode_ms_per_token is not None else "n/a"
        decode_tps = f"{r.decode_tok_per_s:.1f}" if r.decode_tok_per_s is not None else "n/a"
        print(
            f"{r.method:<14}{r.mode:<11}{r.context_prefill_s_mean:>14.3f}{r.ttft_s_mean:>13.3f}"
            f"{r.ttft_s_first:>11.3f}{decode_ms:>15}{decode_tps:>14}"
            f"{r.e2e_tok_per_s:>11.1f}{r.total_wall_s:>9.2f}{r.compression_ratio_context:>13.2%}"
        )
    for r in results:
        memory = ", ".join(f"{device}: {gb:.1f} GB" for device, gb in r.peak_memory_gb.items())
        print(f"peak memory [{r.method}/{r.mode}]: {memory} (KV cache {r.kv_cache_gb:.2f} GB)")


def benchmark(
    model_name: str = "Qwen/Qwen3-8B",
    methods: str | tuple[str, ...] = ("none", "kvzap", "triattention"),
    kvzap_compression_ratio: float = 0.5,
    triattention_stats: Optional[str] = None,
    triattention_compression_ratio: float = 0.5,
    num_gpus: int = 2,
    context_length: int = 16000,
    n_questions: int = 4,
    max_new_tokens: int = 256,
    ignore_eos: bool = True,
    context_file: Optional[str] = None,
    output_file: Optional[str] = None,
    cpu_staging: Optional[bool] = None,
    quality: bool = True,
    divergence_tokens: int = 32,
    samples_file: Optional[str] = None,
    n_contexts: int = 1,
    context_lengths=None,
    compression_ratios=None,
    repeats: int = 1,
    match_tolerance: float = 0.02,
    plot: bool = True,
    seed: int = 0,
):
    """Benchmark compression methods and context cache reuse vs recompute on a layer-sharded model.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model name, by default "Qwen/Qwen3-8B" (needs a matching nvidia/KVzap-* model)
    methods : str or tuple of str, optional
        Methods to compare among "none", "kvzap" and "triattention", by default all of them
    kvzap_compression_ratio : float, optional
        Fraction of context KV pairs physically removed using the KVzap MLP surrogate, by default 0.5
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
    cpu_staging : bool, optional
        Route cross-GPU tensors through CPU memory; by default enabled if a peer-copy integrity check fails
    quality : bool, optional
        Evaluate against reference answers and the dense baseline in separate untimed passes, by default True
    divergence_tokens : int, optional
        Number of dense-greedy continuation positions for teacher-forced KL/JS/NLL/agreement; zero disables probes
    samples_file : str, optional
        JSON/JSONL contexts with question objects containing question, answers, and optional needle_depth
    n_contexts : int, optional
        Maximum contexts selected from the sample file, or number of LooGLE contexts, by default 1
    context_lengths, compression_ratios : str or sequence, optional
        Comma-separated context caps and shared KVzap/TriAttention physical removal ratios for sweeps.
        Without compression_ratios, each method uses its own compression_ratio option.
    repeats : int, optional
        Timed repetitions at each operating point; bootstrap CIs do not count repeats as independent contexts
    match_tolerance : float, optional
        Maximum absolute gap in achieved physical compression for paired comparisons, by default 0.02
    plot : bool, optional
        Write quality/efficiency plots beside the JSON report, by default True
    seed : int, optional
        Torch random seed, by default 0
    """
    methods = methods.split(",") if isinstance(methods, str) else list(methods)
    methods = list(dict.fromkeys(method.strip() for method in methods))
    lengths = parse_sweep(context_lengths, context_length, int)
    kvzap_ratio_values = parse_sweep(compression_ratios, kvzap_compression_ratio)
    ratio_values = parse_sweep(compression_ratios, triattention_compression_ratio)
    if min(n_questions, n_contexts, repeats, max_new_tokens, num_gpus, *lengths) <= 0:
        raise ValueError("Context lengths, counts, max_new_tokens, and num_gpus must be positive")
    if (
        divergence_tokens < 0
        or not 0 <= match_tolerance <= 1
        or any(not 0 <= ratio < 1 for ratio in kvzap_ratio_values + ratio_values)
    ):
        raise ValueError("Need divergence_tokens >= 0, tolerance in [0, 1], compression ratios in [0, 1)")
    if samples_file and context_file:
        raise ValueError("Use either samples_file or context_file, not both")
    if quality and "none" not in methods:
        logger.info("Adding dense baseline for paired quality evaluation")
        methods.insert(0, "none")
    if "none" in methods:
        methods = ["none"] + [method for method in methods if method != "none"]
    if not methods:
        raise ValueError("At least one method is required")
    samples = load_benchmark_samples(context_file, samples_file, n_questions, n_contexts)
    if quality and any(not references for sample in samples for references in sample["references"]):
        logger.warning("Some questions have no references: their answer-quality scores will be unavailable, not zero")
    torch.manual_seed(seed)
    presses = []
    for method in methods:
        if method == "none":
            presses.append((method, "dense", None))
        elif method == "kvzap":
            for value in kvzap_ratio_values:
                presses.append((method, f"ratio={value!r}", KVzapPress(model_type="mlp", compression_ratio=value)))
        elif method == "triattention":
            assert triattention_stats is not None, "--triattention_stats is required for the triattention method"
            for value in ratio_values:
                presses.append(
                    (
                        method,
                        f"ratio={value!r}",
                        TriAttentionPress(compression_ratio=value, stats_path=triattention_stats),
                    )
                )
        else:
            raise ValueError(f"Unknown method {method}, expected none, kvzap or triattention")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = load_sharded_model(model_name, num_gpus=num_gpus, cpu_staging=cpu_staging)
    print(f"Sharded {model_name}: {describe_device_map(model)}")

    # Reuse the pipeline preprocessing (chat template, context truncation) so that the inputs match evaluation
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=tokenizer)
    results, sample_records, input_manifest = [], [], []
    warmed_up = False
    for sample in samples:
        for length in lengths:
            inputs = pipe.preprocess(
                sample["context"], sample["questions"], answer_prefix="", max_context_length=length
            )
            context_ids = inputs["context_ids"].to(model.device)
            questions_ids = [question.to(model.device) for question in inputs["questions_ids"]]
            input_manifest.append(
                {
                    "context_id": sample["id"],
                    "context_length_requested": length,
                    "context_token_ids": context_ids.cpu().tolist()[0],
                    "question_token_ids": [question.cpu().tolist()[0] for question in questions_ids],
                    "questions": sample["questions"],
                }
            )
            print(f"Context {sample['id']}: {context_ids.shape[1]} tokens, {len(questions_ids)} questions")
            if not warmed_up:
                for method, _, press in presses:
                    run(pipe, method, context_ids[:, :1024], questions_ids[:1], press, True, 8, ignore_eos)
                warmed_up = True
            for repetition in range(repeats):
                paired_runs = []
                dense_results = {}
                for method, setting, press in presses:
                    for reuse in [True, False]:
                        result = run(pipe, method, context_ids, questions_ids, press, reuse, max_new_tokens, ignore_eos)
                        result.setting, result.context_id, result.repetition = setting, sample["id"], repetition
                        results.append(result)
                        paired_runs.append((result, press))
                        if method == "none":
                            dense_results[result.mode] = result
                        print(f"done: method={method} {setting} mode={result.mode} total={result.total_wall_s:.2f}s")
                for result, press in paired_runs:
                    measured = (
                        measure_quality(
                            pipe,
                            result,
                            dense_results[result.mode],
                            context_ids,
                            questions_ids,
                            press,
                            sample["references"],
                            divergence_tokens,
                        )
                        if quality
                        else [{} for _ in questions_ids]
                    )
                    for index, (question, measurement) in enumerate(zip(result.questions, measured)):
                        sample_records.append(
                            {
                                **question,
                                **measurement,
                                "method": result.method,
                                "setting": result.setting,
                                "mode": result.mode,
                                "context_id": sample["id"],
                                "context_tokens": result.context_tokens,
                                "context_length_requested": length,
                                "question_index": index,
                                "repetition": repetition,
                                "needle_depth": sample["needle_depths"][index],
                                "needle_depth_basis": "input_metadata_before_truncation",
                                "peak_memory_gb": result.peak_memory_gb,
                                "peak_memory_gb_total": sum(result.peak_memory_gb.values()),
                                "run_e2e_tok_per_s": result.e2e_tok_per_s,
                                "quality_available": bool(sample["references"][index]) and quality,
                            }
                        )

    print()
    print_table(results)

    output_path = Path(output_file or f"results/benchmark_sharded__{model_name.replace('/', '--')}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "model_name": model_name,
        "methods": methods,
        "kvzap_model_type": "mlp",
        "kvzap_compression_ratio": kvzap_compression_ratio,
        "kvzap_compression_ratios": kvzap_ratio_values,
        "kvzap_pruning": "physical_topk",
        "compression_policy": "physical_topk_prefill_only",
        "triattention_stats": triattention_stats,
        "triattention_compression_ratio": triattention_compression_ratio,
        "num_gpus": num_gpus,
        "cpu_staging": model.kvpress_cpu_staging,
        "device_map": describe_device_map(model),
        "max_new_tokens": max_new_tokens,
        "ignore_eos": ignore_eos,
        "quality": quality,
        "divergence_tokens": divergence_tokens,
        "samples_file": samples_file,
        "context_file": context_file,
        "n_contexts": len(samples),
        "n_questions": n_questions,
        "context_lengths": lengths,
        "compression_ratios": ratio_values,
        "repeats": repeats,
        "match_tolerance": match_tolerance,
        "seed": seed,
        "torch_version": torch.__version__,
        "model_dtype": str(model.dtype),
        "gpu_names": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }
    report = {
        "schema_version": 3,
        "config": config,
        "inputs": input_manifest,
        "results": [asdict(result) for result in results],
        "samples": sample_records,
        "summary": summarize_samples(sample_records),
        "matched_compression_pairs": matched_compression_pairs(sample_records, match_tolerance),
        "notes": [
            "KV slots count batch x KV heads x tokens, summed across layers; K and V together form one slot.",
            "KVzap uses an MLP surrogate and physical top-k pruning; "
            "TriAttention also physically prunes context slots.",
            "Achieved and physical compression both count removed context slots; KV bytes are measured from tensors.",
            "Compression is prefill-only, without a protected recent-token window or decoding-time eviction.",
            "Matched pairs match context compression, not KV bytes, peak memory, or latency budgets.",
            "Divergence uses identical dense-greedy prefixes in a separate untimed pass; units are nats.",
            "Peak memory is per timed run, shared by its question records; "
            "summed per-device peaks need not be simultaneous.",
            "CI95 bootstraps independent context IDs; a single context has no CI, regardless of questions or repeats.",
            "Needle depth is supplied metadata, not inferred; truncation can remove an answer-bearing span.",
        ],
    }
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"Results saved to {output_path}")
    if plot and quality:
        from kvzap.benchmark_quality import plot_quality_report

        plot_quality_report(report, output_path.with_suffix(".png"))
    return report


if __name__ == "__main__":
    import fire

    fire.Fire(benchmark)
