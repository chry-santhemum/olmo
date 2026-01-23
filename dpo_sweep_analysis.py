"""Analyze DPO filter sweep results: correlate sum_cosine_similarity with sycophancy scores.

Loads metadata.json from each filtered dataset directory and matches it with
sycophancy evaluation results to compute correlations.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sycophancy_eval.analyze import analyze_feedback_logs


def load_filter_metadata(filter_data_dir: Path) -> dict[str, dict]:
    """Load metadata.json from each subdirectory in filter_data_dir.

    Returns:
        dict mapping run name to metadata
    """
    metadata = {}
    for subdir in sorted(filter_data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        meta_path = subdir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                metadata[subdir.name] = json.load(f)
    return metadata


def load_eval_results(results_dir: Path) -> dict[str, dict]:
    """Load sycophancy evaluation results for each model.

    Scans subdirectories for feedback task results.

    Returns:
        dict mapping model name to feedback analysis results
    """
    results = {}
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        feedback_dir = model_dir / "feedback"
        if feedback_dir.exists():
            results[model_dir.name] = analyze_feedback_logs(feedback_dir)
    return results


def match_filter_to_eval(filter_metadata: dict, eval_results: dict) -> list[dict]:
    """Match filter metadata to evaluation results by name.

    Assumes checkpoint names follow pattern: olmo3_7b_instruct_dpo_{filter_name}

    Returns:
        List of dicts with matched filter config and eval results
    """
    matched = []
    for filter_name, meta in filter_metadata.items():
        checkpoint_name = f"olmo3_7b_instruct_dpo_{filter_name}"
        if checkpoint_name in eval_results:
            matched.append({
                "name": filter_name,
                "sum_cosine_similarity": meta["sum_cosine_similarity"],
                "mean_cosine_similarity": meta["mean_cosine_similarity"],
                "num_samples": meta["num_samples"],
                "num_unique_samples": meta["num_unique_samples"],
                "num_flipped": meta["num_flipped"],
                "filter_config": meta["filter_config"],
                "eval_results": eval_results[checkpoint_name],
            })
    return matched


def compute_correlations(matched_data: list[dict]) -> dict:
    """Compute correlations between filter metrics and sycophancy scores."""
    if len(matched_data) < 3:
        return {"error": "Not enough data points for correlation"}

    x_sum = [d["sum_cosine_similarity"] for d in matched_data]
    x_mean = [d["mean_cosine_similarity"] for d in matched_data]
    y = [d["eval_results"].get("overall_biased_wins_rate", 0) for d in matched_data]

    results = {}

    # Pearson correlation for sum
    if len(set(x_sum)) > 1:
        r_sum, p_sum = stats.pearsonr(x_sum, y)
        results["sum_pearson_r"] = r_sum
        results["sum_pearson_p"] = p_sum

    # Pearson correlation for mean
    if len(set(x_mean)) > 1:
        r_mean, p_mean = stats.pearsonr(x_mean, y)
        results["mean_pearson_r"] = r_mean
        results["mean_pearson_p"] = p_mean

    # Spearman correlation for sum
    if len(set(x_sum)) > 1:
        rho_sum, sp_sum = stats.spearmanr(x_sum, y)
        results["sum_spearman_rho"] = rho_sum
        results["sum_spearman_p"] = sp_sum

    return results


def plot_correlation(matched_data: list[dict], output_path: Path, metric: str = "sum"):
    """Plot scatter plot with regression line for cosine similarity vs sycophancy score."""
    if metric == "sum":
        x = [d["sum_cosine_similarity"] for d in matched_data]
        xlabel = "Sum Cosine Similarity"
    else:
        x = [d["mean_cosine_similarity"] for d in matched_data]
        xlabel = "Mean Cosine Similarity"

    y = [d["eval_results"].get("overall_biased_wins_rate", 0) for d in matched_data]
    names = [d["name"] for d in matched_data]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter plot
    ax.scatter(x, y, s=100, alpha=0.7, edgecolors="black", linewidths=1)

    # Annotate points
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Regression line
    if len(set(x)) > 1:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        x_line = np.linspace(min(x), max(x), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "r--", alpha=0.7, label=f"r={r_value:.3f}, p={p_value:.3f}")
        ax.legend()

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Overall Biased Wins Rate", fontsize=12)
    ax.set_title(f"DPO Filter Sweep: {xlabel} vs Sycophancy Score", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def print_summary_table(matched_data: list[dict]):
    """Print summary table of matched filter configs and results."""
    print("\n" + "=" * 100)
    print("DPO FILTER SWEEP ANALYSIS")
    print("=" * 100)

    headers = ["Name", "Sum Cos Sim", "Mean Cos Sim", "Samples", "Unique", "Flipped", "Syco Rate"]
    widths = [30, 12, 12, 10, 10, 10, 12]

    # Header
    header_line = " | ".join(h.center(w) for h, w in zip(headers, widths))
    print(header_line)
    print("-" * len(header_line))

    # Data rows (sorted by sum cosine similarity)
    for d in sorted(matched_data, key=lambda x: x["sum_cosine_similarity"]):
        syco_rate = d["eval_results"].get("overall_biased_wins_rate", 0)
        row = [
            d["name"][:30],
            f"{d['sum_cosine_similarity']:.2f}",
            f"{d['mean_cosine_similarity']:.4f}",
            str(d["num_samples"]),
            str(d["num_unique_samples"]),
            str(d["num_flipped"]),
            f"{syco_rate:.3f}",
        ]
        print(" | ".join(str(r).ljust(w) for r, w in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description="Analyze DPO filter sweep results")
    parser.add_argument("--filter-data-dir", type=str, default="dpo_filter_data",
                        help="Directory containing filtered dataset directories")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing sycophancy evaluation results")
    parser.add_argument("--output-dir", type=str, default="dpo_sweep_analysis",
                        help="Output directory for plots and analysis")
    args = parser.parse_args()

    filter_data_dir = Path(args.filter_data_dir)
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading filter metadata from: {filter_data_dir}")
    filter_metadata = load_filter_metadata(filter_data_dir)
    print(f"Found {len(filter_metadata)} filter configurations")

    print(f"Loading evaluation results from: {results_dir}")
    eval_results = load_eval_results(results_dir)
    print(f"Found {len(eval_results)} evaluation results")

    # Match filter configs to eval results
    matched_data = match_filter_to_eval(filter_metadata, eval_results)
    print(f"Matched {len(matched_data)} configs to eval results")

    if not matched_data:
        print("Error: No matching data found")
        return

    # Print summary table
    print_summary_table(matched_data)

    # Compute correlations
    correlations = compute_correlations(matched_data)
    print("\nCorrelations:")
    for k, v in correlations.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Generate plots
    plot_correlation(matched_data, output_dir / "sum_cosine_vs_sycophancy.png", metric="sum")
    plot_correlation(matched_data, output_dir / "mean_cosine_vs_sycophancy.png", metric="mean")

    # Save results to JSON
    output_data = {
        "matched_data": [
            {
                "name": d["name"],
                "sum_cosine_similarity": d["sum_cosine_similarity"],
                "mean_cosine_similarity": d["mean_cosine_similarity"],
                "num_samples": d["num_samples"],
                "num_unique_samples": d["num_unique_samples"],
                "num_flipped": d["num_flipped"],
                "filter_config": d["filter_config"],
                "overall_biased_wins_rate": d["eval_results"].get("overall_biased_wins_rate"),
            }
            for d in matched_data
        ],
        "correlations": correlations,
    }
    output_file = output_dir / "sweep_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
