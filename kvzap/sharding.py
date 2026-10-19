# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel


def layer_split_device_map(model_name: str, num_gpus: int = 2) -> dict[str, int]:
    """
    Build a device_map that splits the decoder layers into contiguous, equally sized blocks, one per GPU
    (pipeline-style sharding). The embeddings and rotary embeddings live on the first GPU, the final norm
    on the last GPU. The lm_head goes on the last GPU unless it is tied to the input embeddings.

    Accelerate moves the hidden states (and the position embeddings, cache_position, ...) between GPUs at the
    layer boundaries, so every attention layer and its KV cache stay on the same device. KVzapPress moves each
    per-layer surrogate model to the device of its layer, and DMSPress keeps the masked indices on that device.
    """
    config = AutoConfig.from_pretrained(model_name)
    num_layers = config.num_hidden_layers
    assert torch.cuda.device_count() >= num_gpus, f"Need {num_gpus} GPUs, found {torch.cuda.device_count()}"
    assert num_layers >= num_gpus, f"Cannot split {num_layers} layers across {num_gpus} GPUs"

    device_map: dict[str, int] = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": num_gpus - 1}
    for layer_idx in range(num_layers):
        device_map[f"model.layers.{layer_idx}"] = layer_idx * num_gpus // num_layers
    device_map["lm_head"] = 0 if getattr(config, "tie_word_embeddings", False) else num_gpus - 1
    return device_map


def load_sharded_model(model_name: str, num_gpus: int = 2, dtype: str = "auto") -> PreTrainedModel:
    """Load a causal LM with its decoder layers split across `num_gpus` GPUs."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, device_map=layer_split_device_map(model_name, num_gpus)
    )
    model.eval()
    return model


def describe_device_map(model) -> str:
    """Return a one-line summary of the layer ranges per device, e.g. 'cuda:0: layers 0-17 | cuda:1: layers 18-35'."""
    per_device: dict[str, list[int]] = {}
    for layer_idx, layer in enumerate(model.model.layers):
        per_device.setdefault(str(next(layer.parameters()).device), []).append(layer_idx)
    return " | ".join(f"{device}: layers {idx[0]}-{idx[-1]}" for device, idx in per_device.items())


def synchronize_all():
    """Synchronize every visible GPU (torch.cuda.synchronize only waits for the current one)."""
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.synchronize(device_idx)
