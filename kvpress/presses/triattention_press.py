# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Literal, Optional

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress


@dataclass
class TriAttentionPress(ScorerPress):
    """
    TriAttention (https://arxiv.org/abs/2604.04921) scores keys with a trigonometric series built from
    calibrated per-head statistics of the pre-RoPE queries, instead of using the current queries.

    For every attention head and RoPE frequency f, calibration provides the mean query q_mean_f (complex) and
    the mean query norm E|q_f|. The score of a key k is the expected attention logit of a future query at
    distance d, averaged over the geometric distances d = t + {1, 2, 4, ..., offset_max_length} - p:
        sum_f |q_mean_f| |k_f| cos(omega_f d + arg(q_mean_f) - arg(k_f)) + (E|q_f| - |q_mean_f|) |k_f|
    Since the keys in the cache are already rotated by omega_f p, the average over distances reduces to a single
    product with the rotated keys, so the key positions are not needed (compatible with DecodingPress).
    Scores of the query heads sharing a KV head are aggregated with `head_aggregation`.

    The statistics are the .pt files of the official implementation (https://github.com/WeianMao/triattention),
    e.g. triattention/calibration/for_aime25_experiment/qwen3_8b.pt for Qwen/Qwen3-8B.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    stats_path : str, optional
        Path to the TriAttention calibration statistics (.pt file). Required.
    offset_max_length : int, default=65536
        Largest future query distance used to average the scores (distances are the powers of 2 up to it).
    head_aggregation : {"max", "mean"}, default="max"
        How the scores of the query heads of a KV head are aggregated.
    disable_mlr : bool, default=False
        Use E|q_f| instead of E|q_f| - |q_mean_f| in the position-independent term (ablation from the paper).
    disable_trig : bool, default=False
        Only keep the position-independent term (ablation from the paper).
    """

    stats_path: Optional[str] = None
    offset_max_length: int = 65536
    head_aggregation: Literal["max", "mean"] = "max"
    disable_mlr: bool = False
    disable_trig: bool = False
    q_mean: torch.Tensor = field(init=False, default=None, repr=False)
    q_abs_mean: torch.Tensor = field(init=False, default=None, repr=False)
    inv_freq: torch.Tensor = field(init=False, default=None, repr=False)
    attention_scaling: float = field(init=False, default=1.0, repr=False)

    def post_init_from_model(self, model):
        if self.stats_path is None:
            raise ValueError("TriAttentionPress requires stats_path, the path to the TriAttention calibration stats")
        config = model.config.get_text_config()
        self.q_mean, self.q_abs_mean = self.load_stats(
            self.stats_path, config.num_hidden_layers, config.num_attention_heads
        )
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        self.inv_freq = language_model.rotary_emb.inv_freq.float().cpu()
        self.attention_scaling = float(getattr(language_model.rotary_emb, "attention_scaling", 1.0))

    @staticmethod
    def load_stats(stats_path: str, num_layers: int, num_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load the per-head statistics as (num_layers, num_heads, head_dim // 2) tensors."""
        payload = torch.load(stats_path, map_location="cpu", weights_only=True)
        rope_style = payload["metadata"].get("rope_style", "half")
        assert rope_style == "half", f"Only rope_style='half' is supported, got {rope_style}"
        q_mean, q_abs_mean = [], []
        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                key = f"layer{layer_idx:02d}_head{head_idx:02d}"
                if key not in payload["stats"]:
                    raise ValueError(f"{stats_path} has no statistics for {key}, was it calibrated for this model?")
                stats = payload["stats"][key]
                q_mean.append(torch.complex(stats["q_mean_real"].float(), stats["q_mean_imag"].float()))
                q_abs_mean.append(stats["q_abs_mean"].float())
        return (
            torch.stack(q_mean).view(num_layers, num_heads, -1),
            torch.stack(q_abs_mean).view(num_layers, num_heads, -1),
        )

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        if self.q_mean is None:
            raise ValueError(
                "Stats not loaded. If you are using a wrapper press, make sure to call post_init_from_model."
            )
        bsz, num_kv_heads, k_len, head_dim = keys.shape
        n_freq = head_dim // 2
        device = keys.device
        layer_idx: int = module.layer_idx  # type: ignore[assignment]

        q_mean = self.q_mean[layer_idx].to(device)  # (num_heads, n_freq)
        q_abs_mean = self.q_abs_mean[layer_idx].to(device)
        num_groups = q_mean.shape[0] // num_kv_heads

        # Average of exp(i omega_f (t + offset)) over the future query positions, t being the next position
        t = float(kwargs["cache_position"][-1]) + 1
        offsets = [2**i for i in range(self.offset_max_length.bit_length())]
        offsets_t = torch.tensor(offsets, dtype=torch.float64, device=device)
        omega = self.inv_freq[:n_freq].to(device, torch.float64)
        phases = torch.exp(1j * omega[None, :] * (t + offsets_t[:, None])).mean(dim=0).to(torch.complex64)

        # Rotated keys as complex numbers (RoPE pairs the two halves of the head dimension)
        k = keys.float() / self.attention_scaling
        k = torch.complex(k[..., :n_freq], k[..., n_freq:])  # (bsz, num_kv_heads, k_len, n_freq)
        k_abs = k.abs()

        # Position-independent term: sum_f (E|q_f| - |q_mean_f|) |k_f|
        q_extra = q_abs_mean if self.disable_mlr else q_abs_mean - q_mean.abs()
        scores = k_abs @ q_extra.view(num_kv_heads, num_groups, n_freq).transpose(1, 2)

        # Trigonometric term: Re(sum_f q_mean_f conj(k_f) mean_offset(exp(i omega_f (t + offset))))
        if not self.disable_trig:
            q_trig = (q_mean * phases).view(num_kv_heads, num_groups, n_freq).transpose(1, 2)
            scores = scores + (k.conj() @ q_trig).real

        # RoPE frequency scaling (attention_scaling ** 2 for each frequency, as in the official implementation)
        scores = scores * self.attention_scaling**2  # (bsz, num_kv_heads, k_len, num_groups)

        if self.head_aggregation == "max":
            return scores.amax(dim=-1)
        return scores.mean(dim=-1)
