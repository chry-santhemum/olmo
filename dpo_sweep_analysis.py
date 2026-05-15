"""Analyze seed-averaged DPO ablations with paired bootstrap CIs."""

import json
import re
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TASK_NAME = "feedback_v2"
SFT_MODEL_DIR = Path("sycophancy_eval/results/20260424_033530/olmo-3-7b-instruct-sft")
FILTERED_RESULTS_DIRS = [
    Path("sycophancy_eval/results/20260424_042228"),
    Path("sycophancy_eval/results/20260426_161537"),
    Path("sycophancy_eval/results/20260427_042812"),
    Path("sycophancy_eval/results/20260427_133148"),
    Path("sycophancy_eval/results/20260427_141257"),
    Path("sycophancy_eval/results/20260427_152858"),
    Path("sycophancy_eval/results/20260504_155155"),
    Path("sycophancy_eval/results/20260504_165201"),
]
OUTPUT_PATH = Path("dpo_sweep_analysis/dpo_ablation_bootstrap.png")

BOOTSTRAP_SAMPLES = 2000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 0

CHECKPOINT_PREFIX = "olmo3_7b_instruct_dpo_16K-"
SEED_SUFFIX_PATTERN = re.compile(r"-seed-\d+$")
BIAS_TYPES = ["positive", "negative"]
BIAS_DISPLAY_NAMES = {
    "positive": "positive bias",
    "negative": "negative bias",
}
BAR_WIDTH = 0.42
BAR_SPACING = 0.65
DOT_JITTER_WIDTH = 0.10
X_TICK_ROTATION = 55

SampleKey = tuple[str, str]


@dataclass(frozen=True)
class EvalRun:
    model_name: str
    protocol_name: str
    eval_path: Path
    samples: dict[SampleKey, float]


@dataclass(frozen=True)
class MetricStats:
    mean: float
    ci_low: float
    ci_high: float


def newest_eval_path(model_dir: Path) -> Path:
    eval_paths = list(model_dir.glob(f"{TASK_NAME}/*.eval"))
    if not eval_paths:
        raise FileNotFoundError(f"No {TASK_NAME} eval files found under {model_dir}")
    return max(eval_paths, key=lambda path: path.stat().st_mtime)


def protocol_name_for_model(model_name: str) -> str:
    if model_name == "olmo-3-7b-instruct-dpo":
        return "full-dpo"
    if model_name.startswith(CHECKPOINT_PREFIX):
        model_name = model_name.removeprefix(CHECKPOINT_PREFIX)
    return SEED_SUFFIX_PATTERN.sub("", model_name)


def extract_sycophancy_samples(eval_path: Path) -> dict[SampleKey, float]:
    """Return per-sample scores where larger values mean more sycophancy.

    The raw feedback_v2 score is positive for positive user bias and negative for
    negative user bias, so negative-bias samples are sign-flipped before analysis.
    """
    samples: dict[SampleKey, float] = {}

    with zipfile.ZipFile(eval_path) as eval_zip:
        for member_name in eval_zip.namelist():
            if not (member_name.startswith("samples/") and member_name.endswith(".json")):
                continue

            sample = json.loads(eval_zip.read(member_name))
            sample_metadata = sample.get("metadata")
            scorer = sample.get("scores", {}).get("graded_comparison_scorer")
            if not isinstance(sample_metadata, dict) or not isinstance(scorer, dict):
                raise ValueError(f"Malformed scored sample {member_name} in {eval_path}")

            judge_metadata = scorer.get("metadata")
            if not isinstance(judge_metadata, dict):
                raise ValueError(f"Missing judge metadata for {member_name} in {eval_path}")

            bias_type = judge_metadata.get("bias_type")
            content_text = sample_metadata.get("content_text")
            sycophancy_score = judge_metadata.get("sycophancy_score")
            if bias_type not in {"positive", "negative"} or not isinstance(content_text, str):
                raise ValueError(f"Missing feedback_v2 sample metadata in {member_name}")
            if not isinstance(sycophancy_score, Real):
                raise ValueError(f"Missing sycophancy score in {member_name}")

            key = (bias_type, content_text)
            if key in samples:
                raise ValueError(f"Duplicate sample key {key!r} in {eval_path}")

            score = float(sycophancy_score)
            samples[key] = score if bias_type == "positive" else -score

    if not samples:
        raise ValueError(f"No scored feedback_v2 samples found in {eval_path}")
    return samples


def load_eval_run(model_dir: Path) -> EvalRun:
    eval_path = newest_eval_path(model_dir)
    model_name = model_dir.name
    return EvalRun(
        model_name=model_name,
        protocol_name=protocol_name_for_model(model_name),
        eval_path=eval_path,
        samples=extract_sycophancy_samples(eval_path),
    )


def load_protocol_runs() -> dict[str, list[EvalRun]]:
    model_dirs = []
    for results_dir in FILTERED_RESULTS_DIRS:
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")
        model_dirs.extend(
            sorted(
                path
                for path in results_dir.iterdir()
                if path.is_dir() and (path / TASK_NAME).is_dir()
            )
        )

    runs_by_protocol: dict[str, list[EvalRun]] = defaultdict(list)
    for model_dir in model_dirs:
        run = load_eval_run(model_dir)
        runs_by_protocol[run.protocol_name].append(run)

    return dict(
        sorted(runs_by_protocol.items(), key=lambda item: protocol_sort_key(item[0]))
    )


def protocol_sort_key(protocol_name: str) -> tuple[int, str]:
    return (0, protocol_name) if protocol_name == "full-dpo" else (1, protocol_name)


def protocol_score_matrix(runs: list[EvalRun], sample_keys: list[SampleKey]) -> np.ndarray:
    """Return one row per model seed, aligned to the SFT sample order."""
    expected_keys = set(sample_keys)
    run_scores = []
    for run in runs:
        run_keys = set(run.samples)
        if not expected_keys <= run_keys:
            missing = len(expected_keys - run_keys)
            raise ValueError(
                f"{run.model_name} does not match the SFT eval samples: "
                f"{missing} missing"
            )
        run_scores.append([run.samples[key] for key in sample_keys])
    return np.asarray(run_scores, dtype=float)


def bootstrap_mean_ci(values: np.ndarray) -> MetricStats:
    mean = float(values.mean())
    if len(values) <= 1 or np.allclose(values, values[0]):
        return MetricStats(mean=mean, ci_low=mean, ci_high=mean)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    resample_indices = rng.integers(
        0,
        len(values),
        size=(BOOTSTRAP_SAMPLES, len(values)),
        endpoint=False,
    )
    bootstrap_means = values[resample_indices].mean(axis=1)
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0
    return MetricStats(
        mean=mean,
        ci_low=float(np.quantile(bootstrap_means, alpha)),
        ci_high=float(np.quantile(bootstrap_means, 1.0 - alpha)),
    )


def plot_results(
    protocol_names: list[str],
    delta_stats: dict[str, dict[str, MetricStats]],
    run_delta_means: dict[str, dict[str, list[float]]],
    output_path: Path,
) -> None:
    x_positions = np.arange(len(protocol_names)) * BAR_SPACING
    color_cycle = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"]
    colors = [color_cycle[index % len(color_cycle)] for index in range(len(protocol_names))]

    fig, axes = plt.subplots(
        len(BIAS_TYPES),
        1,
        figsize=(max(9, 0.85 * len(protocol_names)), 7.5),
        squeeze=False,
    )

    for ax, bias_type in zip(axes[:, 0], BIAS_TYPES):
        stats_by_protocol = delta_stats[bias_type]
        means = [stats_by_protocol[name].mean for name in protocol_names]
        yerr = np.asarray(
            [
                [
                    stats_by_protocol[name].mean - stats_by_protocol[name].ci_low
                    for name in protocol_names
                ],
                [
                    stats_by_protocol[name].ci_high - stats_by_protocol[name].mean
                    for name in protocol_names
                ],
            ]
        )
        ax.bar(x_positions, means, width=BAR_WIDTH, color=colors, alpha=0.75)
        ax.errorbar(
            x_positions,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.2,
            capsize=4,
            zorder=3,
        )

        for x_position, protocol_name, color in zip(x_positions, protocol_names, colors):
            run_means = run_delta_means[bias_type][protocol_name]
            offsets = (
                np.linspace(-DOT_JITTER_WIDTH, DOT_JITTER_WIDTH, len(run_means))
                if len(run_means) > 1
                else [0.0]
            )
            ax.scatter(
                x_position + offsets,
                run_means,
                color=color,
                edgecolor="#222222",
                linewidth=0.8,
                s=42,
                zorder=4,
            )

        ax.axhline(0.0, color="#333333", linewidth=1.0, linestyle=":")
        ax.set_title(BIAS_DISPLAY_NAMES[bias_type])
        ax.set_ylabel("sycophancy score - SFT score")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(protocol_names, rotation=X_TICK_ROTATION, ha="right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def format_ci(stats: MetricStats) -> str:
    return f"{stats.mean:+.4f} [{stats.ci_low:+.4f}, {stats.ci_high:+.4f}]"


def main() -> None:
    sft_run = load_eval_run(SFT_MODEL_DIR)
    sample_keys_by_bias = {
        bias_type: sorted(key for key in sft_run.samples if key[0] == bias_type)
        for bias_type in BIAS_TYPES
    }
    sft_scores = {
        bias_type: np.asarray(
            [sft_run.samples[key] for key in sample_keys_by_bias[bias_type]],
            dtype=float,
        )
        for bias_type in BIAS_TYPES
    }

    runs_by_protocol = load_protocol_runs()
    protocol_names = list(runs_by_protocol)
    score_matrices = {
        bias_type: {
            name: protocol_score_matrix(runs, sample_keys_by_bias[bias_type])
            for name, runs in runs_by_protocol.items()
        }
        for bias_type in BIAS_TYPES
    }
    protocol_scores = {
        bias_type: {
            name: scores.mean(axis=0)
            for name, scores in score_matrices[bias_type].items()
        }
        for bias_type in BIAS_TYPES
    }
    run_delta_means = {
        bias_type: {
            name: (scores - sft_scores[bias_type]).mean(axis=1).tolist()
            for name, scores in score_matrices[bias_type].items()
        }
        for bias_type in BIAS_TYPES
    }
    delta_stats = {
        bias_type: {
            name: bootstrap_mean_ci(scores - sft_scores[bias_type])
            for name, scores in protocol_scores[bias_type].items()
        }
        for bias_type in BIAS_TYPES
    }

    plot_results(
        protocol_names=protocol_names,
        delta_stats=delta_stats,
        run_delta_means=run_delta_means,
        output_path=OUTPUT_PATH,
    )

    print(f"SFT eval: {sft_run.eval_path}")
    for bias_type in BIAS_TYPES:
        print(
            f"SFT {BIAS_DISPLAY_NAMES[bias_type]} score: "
            f"{sft_scores[bias_type].mean():.4f}"
        )
    print(
        "Loaded "
        + ", ".join(
            f"{len(sample_keys_by_bias[bias_type])} {bias_type}"
            for bias_type in BIAS_TYPES
        )
        + " paired eval samples."
    )
    print()
    print("Protocol results")
    print("  delta_vs_sft is a paired bootstrap mean with a 95% CI.")
    print("  Higher scores mean more sycophancy.")
    for name in protocol_names:
        runs = runs_by_protocol[name]
        print(f"  - {name} (n={len(runs)})")
        for bias_type in BIAS_TYPES:
            score_mean = float(protocol_scores[bias_type][name].mean())
            print(
                f"      {BIAS_DISPLAY_NAMES[bias_type]}: "
                f"score={score_mean:.4f}, "
                f"delta_vs_sft={format_ci(delta_stats[bias_type][name])}"
            )
        for run_index, run in enumerate(runs):
            run_deltas = ", ".join(
                f"{bias_type}={run_delta_means[bias_type][name][run_index]:+.4f}"
                for bias_type in BIAS_TYPES
            )
            print(f"      {run.model_name}: {run_deltas}")
            print(f"        {run.eval_path}")
    print()
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
