"""Plot feedback_v2 eval metrics against the fraction of datapoints touched."""

import json
import math
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
from matplotlib.lines import Line2D


CHECKPOINT_PREFIX = "olmo3_7b_instruct_dpo_"
TASK_NAME = "feedback_v2"
ADD_TOTAL_ENTRIES = 16384
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 0
FRACTION_SCALE_EXPONENT = math.log10(2.0)
X_AXIS_PADDING = 0.03
COMMON_X_TICKS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
SERIES_ORDER = ["add", "discard", "flip", "baseline", "SFT", "DPO"]

# Edit these before running the file.
RESULTS_DIRS: list[Path] = [
    # Path("sycophancy_eval/results/20260410_160149"),
    # Path("sycophancy_eval/results/20260410_162751"),
    # Path("sycophancy_eval/results/20260410_162819"),
    # Path("sycophancy_eval/results/20260410_170220"),
    Path("sycophancy_eval/results/20260421_143756")
]
REFERENCE_MODEL_NAME = "olmo3_7b_instruct_dpo_16K-feedback-20.0pct-flip"
FILTERED_DIR = Path("filtered")
OUTPUT_PATH = Path("dpo_sweep_analysis/new_1.png")
CONFIDENCE_LEVEL = DEFAULT_CONFIDENCE_LEVEL
BOOTSTRAP_SAMPLES = DEFAULT_BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = DEFAULT_BOOTSTRAP_SEED

ADD_NAME_PATTERN = re.compile(r"(?:^|-)add-(\d+)(?:$|-)")
PCT_ACTION_PATTERN = re.compile(r"-(\d+(?:\.\d+)?)pct-(flip|prune|discard)(?:$|-)")

SERIES_COLORS = {
    "add": "#1f77b4",
    "discard": "#d62728",
    "flip": "#2ca02c",
    "baseline": "#444444",
    "SFT": "#111111",
    "DPO": "#9467bd",
}

MARKER_BY_APPROACH = {
    "vector": "o",
    "autorater": "s",
}

METRIC_NAME_BY_BIAS = {
    "positive": "feedback_v2/positive_likert_score",
    "negative": "feedback_v2/negative_likert_score",
}

SampleKey = tuple[str, str]


@dataclass(frozen=True)
class SweepInfo:
    dataset_name: str | None
    operation: str
    touched_fraction: float
    series_label: str
    source: str
    metadata_path: str | None


@dataclass(frozen=True)
class MetricStats:
    values: list[float]
    mean: float
    ci_low: float | None
    ci_high: float | None


def dataset_name_for_model(model_name: str) -> str | None:
    if model_name.startswith(CHECKPOINT_PREFIX):
        return model_name.removeprefix(CHECKPOINT_PREFIX)
    return None


def sweep_info_from_metadata(dataset_name: str, metadata: dict, metadata_path: Path) -> SweepInfo:
    """Read the sweep type and touched fraction from filtered dataset metadata."""
    filter_config = metadata.get("filter_config", {})
    extra_dataset_size = filter_config.get("extra_dataset_size")
    if extra_dataset_size:
        return SweepInfo(
            dataset_name=dataset_name,
            operation="add",
            touched_fraction=float(extra_dataset_size) / ADD_TOTAL_ENTRIES,
            series_label="add",
            source="filtered metadata",
            metadata_path=str(metadata_path),
        )

    action = filter_config.get("action")
    if action in {"flip", "discard"}:
        original_num_selected = metadata.get("original_num_selected")
        original_dataset_size = metadata.get("original_dataset_size")
        if original_num_selected is None or original_dataset_size in {None, 0}:
            raise ValueError(
                f"Missing original_num_selected/original_dataset_size in {metadata_path}"
            )
        return SweepInfo(
            dataset_name=dataset_name,
            operation=action,
            touched_fraction=float(original_num_selected) / float(original_dataset_size),
            series_label=action,
            source="filtered metadata",
            metadata_path=str(metadata_path),
        )

    if "baseline" in dataset_name:
        return SweepInfo(
            dataset_name=dataset_name,
            operation="baseline",
            touched_fraction=0.0,
            series_label="baseline",
            source="dataset name",
            metadata_path=str(metadata_path),
        )

    raise ValueError(f"Could not determine sweep operation from {metadata_path}")


def infer_sweep_info_from_name(model_name: str) -> SweepInfo | None:
    """Infer sweep metadata from a checkpoint name when filtered metadata is absent."""
    dataset_name = dataset_name_for_model(model_name)
    if dataset_name and "baseline" in dataset_name:
        return SweepInfo(
            dataset_name=dataset_name,
            operation="baseline",
            touched_fraction=0.0,
            series_label="baseline",
            source="checkpoint name",
            metadata_path=None,
        )

    add_match = ADD_NAME_PATTERN.search(model_name)
    if add_match:
        added_entries = int(add_match.group(1))
        return SweepInfo(
            dataset_name=dataset_name,
            operation="add",
            touched_fraction=added_entries / ADD_TOTAL_ENTRIES,
            series_label="add",
            source="checkpoint name",
            metadata_path=None,
        )

    pct_match = PCT_ACTION_PATTERN.search(model_name)
    if pct_match:
        fraction = float(pct_match.group(1)) / 100.0
        raw_operation = pct_match.group(2)
        operation = "discard" if raw_operation in {"prune", "discard"} else raw_operation
        return SweepInfo(
            dataset_name=dataset_name,
            operation=operation,
            touched_fraction=fraction,
            series_label=operation,
            source="checkpoint name",
            metadata_path=None,
        )

    if dataset_name == "16K-all-flip":
        return SweepInfo(
            dataset_name=dataset_name,
            operation="flip",
            touched_fraction=1.0,
            series_label="flip",
            source="checkpoint name",
            metadata_path=None,
        )

    if model_name == "olmo-3-7b-instruct-sft":
        return SweepInfo(
            dataset_name=None,
            operation="reference",
            touched_fraction=0.0,
            series_label="SFT",
            source="model name",
            metadata_path=None,
        )

    if model_name == "olmo-3-7b-instruct-dpo":
        return SweepInfo(
            dataset_name=None,
            operation="reference",
            touched_fraction=0.0,
            series_label="DPO",
            source="model name",
            metadata_path=None,
        )

    return None


def resolve_sweep_info(model_name: str, filtered_dir: Path) -> SweepInfo:
    """Resolve sweep metadata, preferring filtered metadata over name parsing."""
    dataset_name = dataset_name_for_model(model_name)
    if dataset_name is not None:
        metadata_path = filtered_dir / dataset_name / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            return sweep_info_from_metadata(dataset_name, metadata, metadata_path)

    inferred = infer_sweep_info_from_name(model_name)
    if inferred is not None:
        return inferred

    raise ValueError(
        "Could not resolve touch fraction. Expected filtered metadata or a checkpoint name "
        f"that encodes it: {model_name}"
    )


def find_feedback_v2_eval_paths(results_dirs: list[Path]) -> tuple[dict[str, list[Path]], list[str]]:
    """Collect feedback_v2 eval archives for each model under the given results trees."""
    eval_paths_by_model: dict[str, list[Path]] = defaultdict(list)
    warnings = []

    for results_dir in results_dirs:
        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

        for eval_path in results_dir.rglob(f"{TASK_NAME}/*.eval"):
            model_name = eval_path.parent.parent.name
            eval_paths_by_model[model_name].append(eval_path.resolve())

    for model_name, eval_paths in eval_paths_by_model.items():
        if len(eval_paths) > 1:
            warnings.append(
                f"{model_name}: found {len(eval_paths)} {TASK_NAME} evals, using the newest one"
            )

    return eval_paths_by_model, warnings


def newest_path(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def extract_feedback_v2_metric_samples(eval_path: Path) -> dict[str, dict[SampleKey, float]]:
    """Extract per-sample feedback_v2 likert scores keyed by `(bias_type, content_text)`."""
    metric_samples = {
        metric_name: {} for metric_name in METRIC_NAME_BY_BIAS.values()
    }

    with zipfile.ZipFile(eval_path) as eval_zip:
        for member_name in eval_zip.namelist():
            if not (member_name.startswith("samples/") and member_name.endswith(".json")):
                continue

            sample = json.loads(eval_zip.read(member_name))
            sample_metadata = sample.get("metadata")
            scorer = sample.get("scores", {}).get("graded_comparison_scorer")
            if not isinstance(sample_metadata, dict) or not isinstance(scorer, dict):
                continue

            judge_metadata = scorer.get("metadata")
            if not isinstance(judge_metadata, dict):
                continue

            bias_type = judge_metadata.get("bias_type")
            metric_name = METRIC_NAME_BY_BIAS.get(bias_type)
            content_text = sample_metadata.get("content_text")
            sycophancy_score = judge_metadata.get("sycophancy_score")
            if metric_name is None or not isinstance(content_text, str):
                continue
            if not isinstance(sycophancy_score, Real):
                continue

            key = (bias_type, content_text)
            if key in metric_samples[metric_name]:
                raise ValueError(f"Duplicate sample key {key!r} in {eval_path}")
            metric_samples[metric_name][key] = float(sycophancy_score)

    return {metric_name: samples for metric_name, samples in metric_samples.items() if samples}


def paired_bootstrap_mean_ci(
    reference_samples: dict[SampleKey, float],
    model_samples: dict[SampleKey, float],
    confidence_level: float,
    bootstrap_samples: int,
    seed: int,
) -> MetricStats:
    """Estimate a paired bootstrap CI for `model - reference` on shared eval samples."""
    shared_keys = sorted(reference_samples.keys() & model_samples.keys())
    if not shared_keys:
        raise ValueError("No overlapping samples found for paired bootstrap")

    diffs = np.asarray(
        [model_samples[key] - reference_samples[key] for key in shared_keys],
        dtype=float,
    )
    mean = float(np.mean(diffs))
    values = diffs.tolist()
    if len(diffs) <= 1:
        return MetricStats(values=values, mean=mean, ci_low=None, ci_high=None)
    if np.allclose(diffs, diffs[0]):
        return MetricStats(values=values, mean=mean, ci_low=mean, ci_high=mean)

    rng = np.random.default_rng(seed)
    resample_indices = rng.integers(
        0,
        len(diffs),
        size=(bootstrap_samples, len(diffs)),
        endpoint=False,
    )
    bootstrap_means = diffs[resample_indices].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    ci_low = float(np.quantile(bootstrap_means, alpha))
    ci_high = float(np.quantile(bootstrap_means, 1.0 - alpha))
    return MetricStats(values=values, mean=mean, ci_low=ci_low, ci_high=ci_high)


def load_models(
    eval_paths_by_model: dict[str, list[Path]],
    filtered_dir: Path,
) -> list[dict]:
    """Load per-sample metric values for each discovered model."""
    models = []

    for model_name in sorted(eval_paths_by_model):
        eval_path = newest_path(eval_paths_by_model[model_name])
        sweep_info = resolve_sweep_info(model_name, filtered_dir)
        metric_samples = extract_feedback_v2_metric_samples(eval_path)

        if not metric_samples:
            raise ValueError(f"No {TASK_NAME} sample metrics found in {eval_path}")

        models.append(
            {
                "model_name": model_name,
                "dataset_name": sweep_info.dataset_name,
                "approach": approach_for_sweep_info(sweep_info),
                "operation": sweep_info.operation,
                "touched_fraction": sweep_info.touched_fraction,
                "series_label": sweep_info.series_label,
                "source": sweep_info.source,
                "metadata_path": sweep_info.metadata_path,
                "eval_path": eval_path,
                "metric_samples": metric_samples,
            }
        )

    return models


def reference_model(models: list[dict], reference_model_name: str) -> dict:
    """Return the model whose evals should be used as the paired reference."""
    matching_models = [model for model in models if model["model_name"] == reference_model_name]
    if len(matching_models) != 1:
        raise ValueError(
            f"Expected exactly one reference model named {reference_model_name!r}, "
            f"found {len(matching_models)}"
        )
    return matching_models[0]


def reference_display_name(model: dict) -> str:
    """Return a short label for the chosen reference model."""
    if model["series_label"] in {"SFT", "DPO"}:
        return model["series_label"]
    return model["model_name"]


def add_reference_offsets(
    models: list[dict],
    reference_model_name: str,
    confidence_level: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    """Populate each model's metrics with paired `model - reference` offsets."""
    reference_run = reference_model(models, reference_model_name)
    reference_metric_samples = reference_run["metric_samples"]

    for model in models:
        metrics = {}
        for metric_name in sorted(reference_metric_samples):
            if metric_name not in model["metric_samples"]:
                raise ValueError(f"{model['model_name']} is missing metric {metric_name}")
            if model["model_name"] == reference_model_name:
                sample_count = len(reference_metric_samples[metric_name])
                metrics[metric_name] = MetricStats(
                    values=[0.0] * sample_count,
                    mean=0.0,
                    ci_low=0.0,
                    ci_high=0.0,
                )
                continue

            metrics[metric_name] = paired_bootstrap_mean_ci(
                reference_samples=reference_metric_samples[metric_name],
                model_samples=model["metric_samples"][metric_name],
                confidence_level=confidence_level,
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed,
            )
        model["metrics"] = metrics


def metric_names(models: list[dict]) -> list[str]:
    all_metric_names = set()
    for model in models:
        all_metric_names.update(model["metrics"])
    return sorted(all_metric_names)


def fraction_plot_position(fraction: float) -> float:
    if fraction <= 0:
        return 0.0
    return float(fraction ** FRACTION_SCALE_EXPONENT)


def format_fraction(value: float) -> str:
    if value == 0:
        return "0"
    return f"{value:.2g}"


def humanize(text: str) -> str:
    return text.replace("_", " ")


def series_sort_key(series_label: str) -> tuple[int, str]:
    if series_label in SERIES_ORDER:
        return SERIES_ORDER.index(series_label), series_label
    return len(SERIES_ORDER), series_label


def approach_for_sweep_info(sweep_info: SweepInfo) -> str:
    if sweep_info.series_label in {"SFT", "DPO"}:
        return "reference"
    if sweep_info.dataset_name and "autorater" in sweep_info.dataset_name:
        return "autorater"
    return "vector"


def jittered_x_positions(points: list[dict]) -> list[float]:
    """Spread points slightly when several runs share the same touched fraction."""
    positions = [fraction_plot_position(model["touched_fraction"]) for model in points]
    index_groups: dict[float, list[int]] = defaultdict(list)
    for index, model in enumerate(points):
        index_groups[model["touched_fraction"]].append(index)

    jitter_step = 0.012
    for indices in index_groups.values():
        if len(indices) <= 1:
            continue
        base_position = positions[indices[0]]
        if base_position <= 0.0:
            offsets = np.linspace(0.0, jitter_step * (len(indices) - 1), len(indices))
        elif base_position >= 1.0:
            offsets = np.linspace(-jitter_step * (len(indices) - 1), 0.0, len(indices))
        else:
            offsets = np.linspace(
                -jitter_step * (len(indices) - 1) / 2.0,
                jitter_step * (len(indices) - 1) / 2.0,
                len(indices),
            )
        for index, offset in zip(indices, offsets):
            positions[index] = min(1.0, max(0.0, positions[index] + float(offset)))

    return positions


def legend_handles(models: list[dict], reference_model_name: str) -> list[Line2D]:
    """Build legend entries for the plotted series, approaches, and reference line."""
    reference_run = reference_model(models, reference_model_name)
    non_reference_models = [
        model for model in models if model["model_name"] != reference_model_name
    ]
    handles = []

    present_series_labels = sorted(
        {model["series_label"] for model in non_reference_models},
        key=series_sort_key,
    )
    for series_label in present_series_labels:
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=SERIES_COLORS.get(series_label, "#444444"),
                linestyle="",
                label=series_label,
            )
        )

    for approach in ["vector", "autorater"]:
        if any(model["approach"] == approach for model in non_reference_models):
            handles.append(
                Line2D(
                    [0],
                    [0],
                    marker=MARKER_BY_APPROACH[approach],
                    color="#444444",
                    linestyle="",
                    label=approach,
                )
            )

    handles.append(
        Line2D(
            [0],
            [0],
            color=SERIES_COLORS.get(reference_run["series_label"], "#444444"),
            linestyle=":",
            label=f"reference ({reference_display_name(reference_run)})",
        )
    )
    return handles


def plot_metrics(models: list[dict], output_path: Path, reference_model_name: str) -> None:
    """Save one figure with one subplot per metric."""
    names = metric_names(models)
    if not names:
        raise ValueError("No metrics found to plot")

    reference_run = reference_model(models, reference_model_name)
    reference_name = reference_display_name(reference_run)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 5), squeeze=False)
    axes = list(axes[0])

    for ax, metric_name in zip(axes, names):
        points = [model for model in models if metric_name in model["metrics"]]
        sweep_models = [
            model for model in points if model["model_name"] != reference_model_name
        ]
        sweep_models.sort(
            key=lambda model: (
                model["touched_fraction"],
                series_sort_key(model["series_label"]),
                model["model_name"],
            )
        )

        metric = reference_run["metrics"][metric_name]
        if metric.ci_low is not None and metric.ci_high is not None:
            ax.axhspan(
                metric.ci_low,
                metric.ci_high,
                color=SERIES_COLORS.get(reference_run["series_label"], "#444444"),
                alpha=0.12,
                zorder=1,
            )
        ax.axhline(
            metric.mean,
            color=SERIES_COLORS.get(reference_run["series_label"], "#444444"),
            linestyle=":",
            linewidth=1.5,
            zorder=1,
        )

        x_positions = jittered_x_positions(sweep_models)
        for model, x_position in zip(sweep_models, x_positions):
            metric = model["metrics"][metric_name]
            yerr = None
            if metric.ci_low is not None and metric.ci_high is not None:
                yerr = np.array([[metric.mean - metric.ci_low], [metric.ci_high - metric.mean]])

            series_label = model["series_label"]
            color = SERIES_COLORS.get(series_label, "#444444")
            marker = MARKER_BY_APPROACH.get(model["approach"], "o")
            ax.errorbar(
                [x_position],
                [metric.mean],
                yerr=yerr,
                fmt=marker,
                markersize=7,
                capsize=4,
                linewidth=1.2,
                color=color,
                ecolor=color,
                zorder=2,
            )

        ax.set_xticks([fraction_plot_position(value) for value in COMMON_X_TICKS])
        ax.set_xticklabels([format_fraction(value) for value in COMMON_X_TICKS])
        ax.set_xlim(-X_AXIS_PADDING, 1.0 + X_AXIS_PADDING)
        ax.set_xlabel("Fraction of datapoints touched")
        ax.set_ylabel(f"{humanize(metric_name.split('/', 1)[1])} offset vs {reference_name}")
        ax.set_title(f"{humanize(metric_name).replace('/', ': ')} vs {reference_name}")

    fig.legend(
        handles=legend_handles(models, reference_model_name),
        loc="upper center",
        ncol=4,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_summary(
    models: list[dict],
    warnings: list[str],
    output_path: Path,
    reference_model_name: str,
) -> None:
    reference_run = reference_model(models, reference_model_name)
    print(f"Loaded {len(models)} models for task {TASK_NAME}")
    print(
        f"Reference model: {reference_run['model_name']} "
        f"({reference_display_name(reference_run)})"
    )

    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    print("Models:")
    for model in models:
        print(
            "  - "
            f"{model['model_name']}: "
            f"{model['series_label']} at {model['touched_fraction']:.6f} "
            f"from {model['eval_path']}"
        )

    print(f"Saved plot to {output_path}")


def main(
    results_dirs: list[Path] | None = None,
    output_path: Path | None = None,
    reference_model_name: str | None = None,
) -> None:
    selected_results_dirs = RESULTS_DIRS if results_dirs is None else results_dirs
    if not selected_results_dirs:
        raise ValueError("Set RESULTS_DIRS near the top of this file before running it.")

    selected_reference_model_name = (
        REFERENCE_MODEL_NAME if reference_model_name is None else reference_model_name
    )
    results_dirs = [Path(path) for path in selected_results_dirs]
    output_path = Path(OUTPUT_PATH if output_path is None else output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_paths_by_model, warnings = find_feedback_v2_eval_paths(results_dirs)
    if not eval_paths_by_model:
        raise ValueError(f"No {TASK_NAME} eval files found in the provided results directories")

    models = load_models(
        eval_paths_by_model=eval_paths_by_model,
        filtered_dir=FILTERED_DIR,
    )
    if not models:
        raise ValueError("No models could be loaded from the provided results")

    add_reference_offsets(
        models=models,
        reference_model_name=selected_reference_model_name,
        confidence_level=CONFIDENCE_LEVEL,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    plot_metrics(models, output_path, reference_model_name=selected_reference_model_name)
    print_summary(
        models=models,
        warnings=warnings,
        output_path=output_path,
        reference_model_name=selected_reference_model_name,
    )


if __name__ == "__main__":
    main()
