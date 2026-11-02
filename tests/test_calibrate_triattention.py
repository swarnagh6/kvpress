# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.models.qwen3 import modeling_qwen3

from kvpress import TriAttentionPress
from kvzap.calibrate_triattention import QueryStatsCollector


def test_calibration_stats_match_pre_rope_queries(tmp_path, monkeypatch):
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    model = Qwen3ForCausalLM(config).eval()
    # Random q_norm weights, so that missing the norm changes the statistics
    for layer in model.model.layers:
        layer.self_attn.q_norm.weight.data.uniform_(0.5, 2.0)

    # Reference: the queries passed to RoPE in the attention forward, (bsz, num_heads, q_len, head_dim)
    queries = []
    apply_rotary_pos_emb = modeling_qwen3.apply_rotary_pos_emb

    def recording_apply_rotary_pos_emb(q, k, *args, **kwargs):
        queries.append(q.detach().float())
        return apply_rotary_pos_emb(q, k, *args, **kwargs)

    monkeypatch.setattr(modeling_qwen3, "apply_rotary_pos_emb", recording_apply_rotary_pos_emb)

    input_ids = torch.randint(0, config.vocab_size, (1, 50))
    with torch.no_grad(), QueryStatsCollector(model) as collector:
        model.model(input_ids=input_ids)
    path = tmp_path / "stats.pt"
    torch.save(collector.payload({}), path)

    q_mean, q_abs_mean = TriAttentionPress.load_stats(str(path), config.num_hidden_layers, config.num_attention_heads)
    assert len(queries) == config.num_hidden_layers
    for layer_idx, q in enumerate(queries):
        q = torch.complex(q[0, ..., :8], q[0, ..., 8:])  # (num_heads, q_len, n_freq)
        torch.testing.assert_close(q_mean[layer_idx], q.mean(dim=1))
        torch.testing.assert_close(q_abs_mean[layer_idx], q.abs().mean(dim=1))
