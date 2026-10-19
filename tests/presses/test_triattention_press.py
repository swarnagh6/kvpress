# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from kvpress import TriAttentionPress


def reference_scores(q_mean, q_abs_mean, keys_unrotated, positions, t, inv_freq, offsets, scaling):
    """Direct TriAttention formula on pre-RoPE keys at known positions, looping over the future distances."""
    num_heads, n_freq = q_mean.shape
    bsz, num_kv_heads, k_len, _ = keys_unrotated.shape
    k = torch.complex(keys_unrotated[..., :n_freq], keys_unrotated[..., n_freq:])
    scores = torch.zeros(bsz, num_kv_heads, k_len, num_heads // num_kv_heads)
    for head in range(num_heads):
        kv_head, group = divmod(head, num_heads // num_kv_heads)
        k_head = k[:, kv_head]  # (bsz, k_len, n_freq)
        amplitude = q_mean[head].abs() * k_head.abs()
        phi = torch.angle(q_mean[head] * k_head.conj())
        trig = torch.zeros(bsz, k_len)
        for offset in offsets:
            distance = t + offset - positions  # (k_len,)
            trig += (amplitude * torch.cos(distance[:, None] * inv_freq + phi)).sum(-1) / len(offsets)
        extra = ((q_abs_mean[head] - q_mean[head].abs()) * k_head.abs()).sum(-1)
        scores[:, kv_head, :, group] = scaling**2 * (trig + extra)
    return scores.amax(-1)


@pytest.mark.parametrize("scaling", [1.0, 1.5])
def test_triattention_scores_match_reference(scaling):
    torch.manual_seed(0)
    num_heads, num_kv_heads, head_dim, k_len, t = 4, 2, 16, 20, 300
    n_freq = head_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))

    press = TriAttentionPress(offset_max_length=1024)
    press.q_mean = torch.complex(torch.randn(1, num_heads, n_freq), torch.randn(1, num_heads, n_freq))
    press.q_abs_mean = press.q_mean.abs() + torch.rand(1, num_heads, n_freq)
    press.inv_freq, press.attention_scaling = inv_freq, scaling

    # Keys at non-contiguous positions (as after a compression), rotated with RoPE like in the cache
    keys = torch.randn(1, num_kv_heads, k_len, head_dim)
    positions = torch.arange(k_len).float() * 7 + 3
    angles = positions[:, None] * inv_freq
    cos, sin = angles.cos().repeat(1, 2) * scaling, angles.sin().repeat(1, 2) * scaling
    rotated = keys * cos + torch.cat([-keys[..., n_freq:], keys[..., :n_freq]], dim=-1) * sin

    module = SimpleNamespace(layer_idx=0)
    scores = press.score(module, None, rotated, None, None, {"cache_position": torch.tensor([t - 1])})
    expected = reference_scores(
        press.q_mean[0], press.q_abs_mean[0], keys, positions, t, inv_freq, [2**i for i in range(11)], scaling
    )
    assert scores.shape == (1, num_kv_heads, k_len)
    torch.testing.assert_close(scores, expected, rtol=1e-3, atol=1e-3)


def test_triattention_loads_stats(tmp_path):
    num_layers, num_heads, n_freq = 2, 3, 4
    stats = {
        f"layer{layer:02d}_head{head:02d}": {
            "q_mean_real": torch.full((n_freq,), float(layer)),
            "q_mean_imag": torch.full((n_freq,), float(head)),
            "q_abs_mean": torch.ones(n_freq),
        }
        for layer in range(num_layers)
        for head in range(num_heads)
    }
    path = tmp_path / "stats.pt"
    torch.save({"metadata": {"rope_style": "half"}, "stats": stats}, path)

    q_mean, q_abs_mean = TriAttentionPress.load_stats(str(path), num_layers, num_heads)
    assert q_mean.shape == q_abs_mean.shape == (num_layers, num_heads, n_freq)
    assert q_mean[1, 2, 0] == torch.complex(torch.tensor(1.0), torch.tensor(2.0))

    with pytest.raises(ValueError, match="layer02_head00"):
        TriAttentionPress.load_stats(str(path), num_layers + 1, num_heads)
