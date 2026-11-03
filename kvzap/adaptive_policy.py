"""Bridge the adaptive PoC policy to physical, prefill-only KVzap MLP pruning."""

import hashlib
import importlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch

from kvpress import KVzapPress


def load_policy_modules(policy_root: Optional[str] = None):
    root = (
        Path(policy_root).expanduser().resolve()
        if policy_root
        else Path(__file__).resolve().parents[2] / "adaptive_kvcomp_poc"
    )
    source = root / "adaptive_kvpress_poc" / "policies" / "compression.py"
    if not source.is_file():
        raise FileNotFoundError(
            f"Adaptive policy not found: {source}. Set --adaptive_policy_root to the directory "
            "containing adaptive_kvpress_poc."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    modules = {
        name: importlib.import_module(f"adaptive_kvpress_poc.policies.{name}")
        for name in ("compression", "surrogate", "threshold", "window")
    }
    if Path(modules["compression"].__file__).resolve() != source.resolve():
        raise ValueError("A different adaptive policy checkout is already imported; start a fresh process")
    if not hasattr(modules["compression"].CompressionPolicy, "decide_scores"):
        raise ValueError("Adaptive policy needs the real-score CompressionPolicy.decide_scores API")
    return modules


def prefix_block_hashes(token_ids: list[int], block_size: int = 512) -> list[int]:
    """Hash causal prefixes, not isolated equal blocks with different histories."""
    digest = hashlib.sha256()
    hashes = []
    for start in range(0, len(token_ids), block_size):
        digest.update(json.dumps(token_ids[start : start + block_size], separators=(",", ":")).encode("ascii"))
        hashes.append(int.from_bytes(digest.digest()[:8], "big"))
    return hashes


def prepare_reuse_metadata(inputs: list[dict]) -> dict[int, Counter]:
    """Count planned question requests independently for each context-length sweep."""
    frequencies = {}
    for entry in inputs:
        entry["prefix_block_hashes"] = prefix_block_hashes(entry["context_token_ids"])
        frequency = frequencies.setdefault(entry["context_length_requested"], Counter())
        for block_hash in entry["prefix_block_hashes"]:
            frequency[block_hash] += len(entry["question_token_ids"])
    return frequencies


@dataclass
class AdaptiveKVzapPress(KVzapPress):
    """Use real MLP scores; one exact, shared token mask per layer and batch."""

    policy_root: Optional[str] = None
    threshold_mode: str = "adaptive"
    tau: float = -1.0
    entropy_sensitivity: float = 0.15
    task_type: str = "general"
    latency_budget_ms: float = 100.0
    base_window: int = 128
    head_tokens: int = 4
    max_window: int = 512
    boundary_extension: int = 64
    scale_window: bool = True
    score_aware_boundary: bool = True
    reuse_aware: bool = True
    reuse_bonus_scale: float = 0.25
    decisions: list[dict] = field(default_factory=list, init=False)
    hash_ids: list[int] = field(default_factory=list, init=False)
    block_frequency: dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.model_type != "mlp":
            raise ValueError("Adaptive benchmark supports only the KVzap MLP scorer")
        if self.threshold_mode not in ("adaptive", "fixed", "percentile"):
            raise ValueError("adaptive_threshold must be adaptive, fixed, or percentile")
        if self.threshold_mode == "adaptive" and self.compression_ratio > 0.75:
            raise ValueError("The PoC adaptive threshold is calibrated only up to a 4x target (0.75 removal)")
        if min(self.base_window, self.head_tokens, self.max_window, self.boundary_extension) < 0:
            raise ValueError("Window sizes must be nonnegative")
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.tau,
                    self.entropy_sensitivity,
                    self.latency_budget_ms,
                    self.reuse_bonus_scale,
                )
            )
            or min(self.entropy_sensitivity, self.latency_budget_ms, self.reuse_bonus_scale) < 0
        ):
            raise ValueError("Adaptive parameters must be finite; sensitivities and budgets must be nonnegative")
        modules = load_policy_modules(self.policy_root)
        self.policy_root = str(Path(modules["compression"].__file__).resolve().parents[2])
        self.policy_sources = {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            }
            for name, module in modules.items()
        }
        threshold = modules["threshold"]
        if self.threshold_mode == "fixed":
            threshold_policy = threshold.FixedThresholdPolicy(self.tau)
        elif self.threshold_mode == "percentile":
            threshold_policy = threshold.PercentileThresholdPolicy(self.compression_ratio)
        else:
            threshold_policy = threshold.AdaptiveThresholdPolicy(
                target_compression=1 / (1 - self.compression_ratio),
                entropy_sensitivity=self.entropy_sensitivity,
            )
        self.policy = modules["compression"].CompressionPolicy(
            threshold_policy=threshold_policy,
            surrogate_selector=modules["surrogate"].SurrogateSelector(modules["surrogate"].SurrogateTier.MLP),
            window_policy=modules["window"].AdaptiveWindowPolicy(
                modules["window"].WindowConfig(
                    base_window=self.base_window,
                    head_tokens=self.head_tokens,
                    max_window=self.max_window,
                    boundary_extension=self.boundary_extension,
                    scale_with_compression=self.scale_window,
                    score_aware_boundary=self.score_aware_boundary,
                    task_type=self.task_type,
                )
            ),
            score_engine=None,
            reuse_aware=self.reuse_aware,
            reuse_bonus_scale=self.reuse_bonus_scale,
        )

    def post_init_from_model(self, model):
        self.decisions = []
        super().post_init_from_model(model)

    def set_workload(self, hash_ids: list[int], block_frequency: dict[int, int]):
        self.hash_ids = list(hash_ids)
        self.block_frequency = dict(block_frequency)

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if keys.shape[0] != 1:
            raise ValueError("Adaptive benchmark supports batch_size=1")
        if self.compression_ratio == 0:
            return keys, values
        from adaptive_kvpress_poc.traces.loader import TraceEntry

        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)
        observed = scores[0].float().mean(dim=0).cpu().tolist()
        entry = TraceEntry(0.0, len(observed), 0, self.hash_ids)
        decision = self.policy.decide_scores(
            observed,
            entry,
            self.block_frequency,
            self.task_type,
            self.latency_budget_ms,
            target_compression=1 / (1 - self.compression_ratio),
        )
        indices = torch.tensor(
            [index for index, keep in enumerate(decision.keep_mask) if keep],
            device=keys.device,
            dtype=torch.long,
        )
        empty_mask_fallback = indices.numel() == 0
        if empty_mask_fallback:
            # An empty layer cache breaks attention: keep the highest-scoring token instead
            indices = torch.tensor([max(range(len(observed)), key=observed.__getitem__)], device=keys.device)
        metadata = asdict(decision)
        del metadata["keep_mask"]
        metadata.update(
            layer=module.layer_idx,
            input_tokens=len(observed),
            tokens_retained=len(indices),
            compression_factor=len(observed) / len(indices),
            physical_removal_fraction=1 - len(indices) / len(observed),
            score_aggregation="mean_over_kv_heads_per_layer",
            scorer="kvzap_mlp",
            empty_mask_fallback=empty_mask_fallback,
            reuse_bonus_applied=bool(
                self.reuse_aware
                and self.reuse_bonus_scale > 0
                and any(
                    self.block_frequency.get(block_hash, 1) > 1
                    for block_hash in self.hash_ids[: (len(observed) + 511) // 512]
                )
            ),
        )
        self.decisions.append(metadata)
        return keys.index_select(2, indices).contiguous(), values.index_select(2, indices).contiguous()
