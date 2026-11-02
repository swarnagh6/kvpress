# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Calibrate the TriAttention statistics (https://arxiv.org/abs/2604.04921) used by TriAttentionPress, for models
without released statistics (e.g. Qwen/Qwen3-32B), with the decoder layers optionally split across several GPUs.

A single forward pass is run on a long plain text. For every attention head and RoPE frequency f, the pre-RoPE
queries q_f (complex, the two halves of the head dimension being the real and imaginary parts) are averaged over
the tokens to get q_mean_f = E[q_f] and E|q_f|. The output has the format of the official .pt files
(https://github.com/WeianMao/triattention) and can be passed to TriAttentionPress(stats_path=...).

The queries are captured right before RoPE: for models with a per-head query norm (q_norm, e.g. Qwen3), after
this norm, so that the statistics match the queries the keys in the cache are attended by. Note that the official
scripts/calibrate.py captures the q_proj output, before q_norm.

Usage (from the repository root, on 2 x A100 80GB):
    python -m kvzap.calibrate_triattention --model_name Qwen/Qwen3-32B --num_gpus 2 \
        --output_file calibration/qwen3_32b.pt
"""

from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from kvzap.sharding import describe_device_map, load_sharded_model

CALIBRATION_SUBSETS = ["Nemotron-CC-High-Quality", "Nemotron-CC-MATH", "Nemotron-Synthetic-Code"]


def load_calibration_ids(
    tokenizer: PreTrainedTokenizerBase, max_length: int, text_file: Optional[str] = None
) -> torch.Tensor:
    """
    Tokenize `text_file`, or documents of the nvidia/Nemotron-Pretraining-Dataset-sample dataset (web, math and
    code, cycling through the subsets), into a single sequence of at most `max_length` tokens.
    """
    if text_file is not None:
        text = Path(text_file).read_text(encoding="utf-8")
    else:
        subsets = [
            load_dataset("nvidia/Nemotron-Pretraining-Dataset-sample", subset, split="train")
            for subset in CALIBRATION_SUBSETS
        ]
        documents, n_tokens = [], 0
        for idx in range(max(len(s) for s in subsets)):
            for dataset in subsets:
                if idx < len(dataset) and n_tokens < max_length:
                    documents.append(dataset[idx]["text"])
                    n_tokens += len(tokenizer.encode(documents[-1], add_special_tokens=False))
            if n_tokens >= max_length:
                break
        text = "\n\n".join(documents)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    return input_ids[:, :max_length]


class QueryStatsCollector:
    """
    Forward pre-hooks on the attention modules accumulating, per head, the sums of the pre-RoPE queries (as complex
    numbers) and of their norms over the tokens. Only these sums are kept, not the queries.
    """

    def __init__(self, model: nn.Module):
        config = model.config.get_text_config()
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.attention_modules = [layer.self_attn for layer in language_model.layers]
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        self.q_sum: dict[int, torch.Tensor] = {}
        self.q_abs_sum: dict[int, torch.Tensor] = {}
        self.n_tokens = 0
        self.hooks: list = []

    def hook(self, module: nn.Module, args, kwargs):
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        # Same as the attention forward before apply_rotary_pos_emb: (bsz, q_len, num_heads, head_dim)
        queries = module.q_proj(hidden_states).view(*hidden_states.shape[:-1], -1, self.head_dim)
        if hasattr(module, "q_norm"):
            queries = module.q_norm(queries)
        queries = queries.float()
        n_freq = self.head_dim // 2
        q = torch.complex(queries[..., :n_freq], queries[..., n_freq:]).flatten(0, 1)  # (tokens, heads, n_freq)
        layer_idx = module.layer_idx
        q_sum, q_abs_sum = q.sum(dim=0).cpu(), q.abs().sum(dim=0).cpu()
        if layer_idx in self.q_sum:
            self.q_sum[layer_idx] += q_sum
            self.q_abs_sum[layer_idx] += q_abs_sum
        else:
            self.q_sum[layer_idx], self.q_abs_sum[layer_idx] = q_sum, q_abs_sum
        if layer_idx == 0:
            self.n_tokens += q.shape[0]

    def __enter__(self):
        self.hooks = [m.register_forward_pre_hook(self.hook, with_kwargs=True) for m in self.attention_modules]
        return self

    def __exit__(self, *exc):
        for hook in self.hooks:
            hook.remove()

    def payload(self, metadata: dict) -> dict:
        """Statistics in the format of the official TriAttention .pt files (see TriAttentionPress.load_stats)."""
        stats = {}
        for layer_idx in sorted(self.q_sum):
            q_mean = self.q_sum[layer_idx] / self.n_tokens
            q_abs_mean = self.q_abs_sum[layer_idx] / self.n_tokens
            for head_idx in range(q_mean.shape[0]):
                stats[f"layer{layer_idx:02d}_head{head_idx:02d}"] = {
                    "q_mean_real": q_mean[head_idx].real.contiguous(),
                    "q_mean_imag": q_mean[head_idx].imag.contiguous(),
                    "q_abs_mean": q_abs_mean[head_idx].contiguous(),
                }
        metadata = {"head_dim": self.head_dim, "rope_style": "half", "num_tokens": self.n_tokens, **metadata}
        return {"metadata": metadata, "stats": stats}


@torch.inference_mode()
def calibrate(
    model_name: str = "Qwen/Qwen3-32B",
    output_file: Optional[str] = None,
    num_gpus: int = 2,
    max_length: int = 32768,
    text_file: Optional[str] = None,
    cpu_staging: Optional[bool] = None,
    attn_implementation: Optional[str] = None,
    local_files_only: bool = False,
):
    """Compute the TriAttention statistics of `model_name` and save them to `output_file`.

    Parameters
    ----------
    model_name : str, optional
        HuggingFace model name, by default "Qwen/Qwen3-32B"
    output_file : str, optional
        Where to save the statistics, by default calibration/<model>.pt
    num_gpus : int, optional
        Number of GPUs to split the decoder layers across (1 to load the model on cuda:0), by default 2
    max_length : int, optional
        Number of calibration tokens, by default 32768 (as the official calibration)
    text_file : str, optional
        Plain text file to calibrate on, by default documents of nvidia/Nemotron-Pretraining-Dataset-sample
    cpu_staging : bool, optional
        Force CPU-staged cross-GPU transfers for hosts with unreliable peer copies
    attn_implementation : str, optional
        Transformers attention implementation, such as flash_attention_2
    local_files_only : bool, optional
        Load the base model and tokenizer only from the local cache
    """
    model_kwargs = {"local_files_only": local_files_only}
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    if num_gpus > 1:
        model = load_sharded_model(model_name, num_gpus=num_gpus, cpu_staging=cpu_staging, **model_kwargs)
        print(f"Sharded {model_name}: {describe_device_map(model)}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="cuda:0", **model_kwargs
        ).eval()

    input_ids = load_calibration_ids(tokenizer, max_length, text_file).to(model.device)
    print(f"Calibrating on {input_ids.shape[1]} tokens")

    with QueryStatsCollector(model) as collector:
        # The decoder only (no lm_head): the logits of all the tokens are not needed
        model.model(input_ids=input_ids, use_cache=False)

    payload = collector.payload(
        {"model_name": model_name, "dtype": str(model.dtype).replace("torch.", ""), "text_file": text_file}
    )
    output_path = Path(output_file or f"calibration/{model_name.split('/')[-1].lower()}.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved the statistics of {len(payload['stats'])} heads to {output_path}")


if __name__ == "__main__":
    import fire

    fire.Fire(calibrate)
