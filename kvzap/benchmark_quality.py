"""Quality and distribution metrics for paired KV-cache benchmarks."""

from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from evaluation.benchmarks.infinite_bench.calculate_metrics import normalize_answer, qa_f1_score


def answer_quality(prediction: str, references: list[str]) -> dict:
    if not references:
        return {}
    normalized = normalize_answer(prediction)
    labels = [normalize_answer(reference) for reference in references]
    return {
        "exact_match": float(any(normalized == label for label in labels)),
        "token_f1": float(qa_f1_score(prediction, references)),
        "answer_containment": float(any(label and f" {label} " in f" {normalized} " for label in labels)),
    }


def token_divergence(reference_logits: torch.Tensor, candidate_logits: torch.Tensor, token_ids: torch.Tensor) -> dict:
    """Compare next-token distributions on an identical teacher-forced prefix, in nats."""
    if reference_logits.shape != candidate_logits.shape or reference_logits.ndim != 2:
        raise ValueError("Logits must have matching [tokens, vocabulary] shapes")
    if token_ids.ndim != 1 or token_ids.shape[0] != reference_logits.shape[0] or not token_ids.numel():
        raise ValueError("One target token is required for every logits row")
    reference_log_probs = reference_logits.float().log_softmax(-1)
    candidate_log_probs = candidate_logits.to(reference_logits.device).float().log_softmax(-1)
    midpoint = torch.logaddexp(reference_log_probs, candidate_log_probs) - np.log(2)
    reference_probs = reference_log_probs.exp()
    candidate_probs = candidate_log_probs.exp()
    reference_nll = -reference_log_probs.gather(-1, token_ids.to(reference_logits.device)[:, None]).squeeze(-1)
    candidate_nll = -candidate_log_probs.gather(-1, token_ids.to(reference_logits.device)[:, None]).squeeze(-1)
    metrics = {
        "kl_dense_to_compressed": (reference_probs * (reference_log_probs - candidate_log_probs)).sum(-1).clamp_min(0),
        "js_divergence": (
            0.5 * (reference_probs * (reference_log_probs - midpoint)).sum(-1)
            + 0.5 * (candidate_probs * (candidate_log_probs - midpoint)).sum(-1)
        ).clamp_min(0),
        "top1_agreement": (reference_log_probs.argmax(-1) == candidate_log_probs.argmax(-1)).float(),
        "dense_nll": reference_nll,
        "compressed_nll": candidate_nll,
        "nll_delta": candidate_nll - reference_nll,
    }
    return {name: values.cpu().tolist() for name, values in metrics.items()}


def distribution_summary(values: list[float], clusters: list[str] | None = None, bootstrap_samples: int = 1000) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Summary values must be finite")
    summary = {
        "n": len(values),
        "mean": float(array.mean()),
        "p5": float(np.quantile(array, 0.05)),
        "min": float(array.min()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }
    if clusters is None:
        clusters = [str(index) for index in range(len(values))]
    if len(clusters) != len(values):
        raise ValueError("Each value needs a cluster ID")
    grouped = {}
    for cluster, value in zip(clusters, values):
        grouped.setdefault(cluster, []).append(value)
    cluster_values = list(grouped.values())
    summary["n_contexts"] = len(cluster_values)
    summary["ci95_mean"] = None
    if len(cluster_values) > 1 and bootstrap_samples > 0:
        sums = np.asarray([sum(group) for group in cluster_values])
        counts = np.asarray([len(group) for group in cluster_values])
        draws = np.random.default_rng(0).integers(len(grouped), size=(bootstrap_samples, len(grouped)))
        estimates = sums[draws].sum(1) / counts[draws].sum(1)
        summary["ci95_mean"] = np.quantile(estimates, [0.025, 0.975]).tolist()
    return summary


def summarize_samples(samples: list[dict]) -> dict:
    breakdowns = {}
    for dimension in ("context_tokens", "needle_depth"):
        values = sorted({sample[dimension] for sample in samples if sample.get(dimension) is not None})
        breakdowns[dimension] = [
            {
                "value": value,
                "groups": summarize_samples_group_only([row for row in samples if row.get(dimension) == value]),
            }
            for value in values
        ]
    return {"groups": summarize_samples_group_only(samples), "breakdowns": breakdowns}


def summarize_samples_group_only(samples: list[dict]) -> list[dict]:
    groups = {}
    for row in samples:
        groups.setdefault((row["method"], row["setting"], row["mode"]), []).append(row)
    output = []
    for (method, setting, mode), rows in groups.items():
        metrics = {}
        names = sorted({name for row in rows for name, value in row.items() if isinstance(value, (int, float))})
        for name in names:
            selected = [row for row in rows if isinstance(row.get(name), (int, float)) and np.isfinite(row[name])]
            metrics[name] = distribution_summary(
                [row[name] for row in selected], [row["context_id"] for row in selected]
            )
        output.append({"method": method, "setting": setting, "mode": mode, "metrics": metrics})
    return output


def matched_compression_pairs(samples: list[dict], tolerance: float = 0.02) -> list[dict]:
    """Return observed, paired operating points; do not interpolate quality or claim matching byte budgets."""
    output = []
    groups = {}
    for row in samples:
        key = (
            row["context_id"],
            row["question_index"],
            row["repetition"],
            row["mode"],
            row.get("context_tokens"),
            row.get("context_length_requested"),
        )
        groups.setdefault(key, []).append(row)
    for rows in groups.values():
        for left, right in combinations(rows, 2):
            if left["method"] == right["method"]:
                continue
            difference = abs(left["achieved_compression_ratio"] - right["achieved_compression_ratio"])
            if difference <= tolerance:
                output.append({"compression_gap": difference, "left": left, "right": right})
    return output


def plot_quality_report(report: dict, output_path):
    """Render quality and its accompanying efficiency on the same achieved-compression axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = report["samples"]
    physical_pruning = report.get("config", {}).get("compression_policy") == "physical_topk_prefill_only"
    compression_label = (
        "Achieved context compression (physical slots)" if physical_pruning else "Achieved context compression"
    )
    metrics = [
        ("token_f1", "Ground-truth token F1", 1),
        ("kl_dense_to_compressed", "Dense-to-compressed KL (nats)", 1),
        ("kv_bytes", "Physical KV cache (GB)", 1e-9),
        ("peak_memory_gb_total", "Sum of per-device peak allocations (GB)", 1),
        ("ttft_s", "TTFT (seconds)", 1),
        ("decode_tok_per_s", "Decode throughput (tokens/s)", 1),
    ]
    figures = []
    for mode in sorted({row["mode"] for row in samples}):
        figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
        for axis, (metric, title, scale) in zip(axes.flat, metrics):
            for method in sorted({row["method"] for row in samples}):
                points = []
                for setting in sorted({row["setting"] for row in samples if row["method"] == method}):
                    rows = [
                        row
                        for row in samples
                        if row["method"] == method
                        and row["setting"] == setting
                        and row["mode"] == mode
                        and row.get(metric) is not None
                    ]
                    if not rows:
                        continue
                    values = [row[metric] * scale for row in rows]
                    stats = distribution_summary(values, [row["context_id"] for row in rows])
                    points.append((np.mean([row["achieved_compression_ratio"] for row in rows]), stats))
                points.sort(key=lambda point: point[0])
                if not points:
                    continue
                compression = [point[0] for point in points]
                axis.plot(compression, [point[1]["mean"] for point in points], "o-", label=method)
                intervals = [point[1]["ci95_mean"] for point in points]
                if all(interval is not None for interval in intervals):
                    axis.fill_between(
                        compression,
                        [interval[0] for interval in intervals],
                        [interval[1] for interval in intervals],
                        alpha=0.12,
                    )
                if metric == "token_f1":
                    axis.plot(compression, [point[1]["p5"] for point in points], ":", label=f"{method} p5")
                    axis.plot(compression, [point[1]["min"] for point in points], "x", label=f"{method} worst")
            axis.set(title=title, xlabel=compression_label, xlim=(-0.02, 1.02))
            axis.grid(alpha=0.2)
            if axis.lines:
                axis.legend(fontsize="small")
        policy_label = "; physical context pruning" if physical_pruning else ""
        figure.suptitle(f"{mode}: observed operating points{policy_label}")
        path = Path(output_path)
        path = path.with_name(f"{path.stem}__{mode}{path.suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160)
        plt.close(figure)
        figures.append(str(path))
    return figures


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Plot paired quality and efficiency from a sharded benchmark report")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    plot_quality_report(json.loads(args.report.read_text()), args.report.with_suffix(".png"))
