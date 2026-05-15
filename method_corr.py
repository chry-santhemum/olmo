import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm


DEFAULT_CACHE_DIR = Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23")
DEFAULT_AUTORATER_PATH = Path("filtered/33K-baseline/dataset_autorated.jsonl")
DEFAULT_LLS_PATH = Path("cached_repr_33K_SFT/LLS/full/scores.jsonl")
DEFAULT_PERSONA_VECTOR_PATH = Path("vectors/Olmo-3-7B-Instruct-SFT/sycophantic_response_avg_diff.pt")
DEFAULT_FEEDBACK_VECTOR_PATH = Path("vectors/feedback_v2_contrast_20260504_051537/contrast_L23.pt")
DEFAULT_OUTPUT_JSON_PATH = Path("method_corr_results.json")
DEFAULT_HEATMAP_PATH = Path("method_corr_heatmap.png")


def load_vector(path: Path, layer: int) -> torch.Tensor:
    """Load either a plain layer-stacked persona vector or a feedback-vector checkpoint."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "layer" in data and data["layer"] != layer:
            raise ValueError(f"{path} stores layer {data['layer']}, but requested layer {layer}")
        return data["vector"].float()

    if not isinstance(data, torch.Tensor):
        raise ValueError(f"Expected {path} to contain a tensor or a dict with a vector")
    if data.ndim == 1:
        return data.float()

    vector_index = layer + 1
    if vector_index >= data.shape[0]:
        raise ValueError(f"Layer {layer} maps to vector index {vector_index}, but {path} has {data.shape[0]} rows")
    return data[vector_index].float()


def load_autorater_scores(path: Path, limit: int | None) -> np.ndarray:
    scores = []
    with open(path) as f:
        for row_index, line in enumerate(f):
            if limit is not None and row_index >= limit:
                break
            row = json.loads(line)
            score = row.get("autorater_score")
            if score is None:
                scores.append(np.nan)
            elif score in {1, 2, 3, 4, 5}:
                scores.append(float(score))
            else:
                raise ValueError(f"Invalid autorater_score at row {row_index}: {score}")
    return np.array(scores, dtype=float)


def load_lls_scores(path: Path, expected_len: int) -> np.ndarray:
    scores = []
    with open(path) as f:
        for row_index, line in enumerate(f):
            if row_index >= expected_len:
                break
            row = json.loads(line)
            if row.get("row_index") != row_index:
                raise ValueError(f"Expected LLS row_index={row_index}, got {row.get('row_index')}")
            scores.append(np.nan if row.get("skip", False) else float(row["score"]))
    if len(scores) < expected_len:
        raise ValueError(f"{path} has {len(scores)} rows, but expected at least {expected_len}")
    return np.array(scores, dtype=float)


def compute_vector_scores(cache_dir: Path, vector: torch.Tensor, layer: int, limit: int | None) -> np.ndarray:
    """Compute cosine similarity between cached chosen-minus-rejected activations and one method vector."""
    with open(cache_dir / "manifest.json") as f:
        manifest = json.load(f)

    chunk_size = manifest["chunk_size"]
    total_chunks = manifest["total_chunks"]
    num_digits = len(str(total_chunks - 1))
    scores = []

    for chunk_idx in tqdm(range(total_chunks), desc=f"Scoring {vector.shape[0]}d vector"):
        if limit is not None and len(scores) >= limit:
            break

        chunk_path = cache_dir / f"chunk_{chunk_idx:0{num_digits}d}.pt"
        chunk = torch.load(chunk_path, map_location="cpu")
        chunk_diffs = chunk["activation_diffs"]
        if chunk_idx < total_chunks - 1 and len(chunk_diffs) != chunk_size:
            raise ValueError(f"Expected chunk {chunk_idx} to have {chunk_size} rows, got {len(chunk_diffs)}")

        remaining = None if limit is None else limit - len(scores)
        if remaining is not None:
            chunk_diffs = chunk_diffs[:remaining]

        diffs = torch.stack([sample[layer].float() for sample in chunk_diffs])
        chunk_scores = F.cosine_similarity(diffs, vector.unsqueeze(0), dim=1).tolist()
        scores.extend(float(score) for score in chunk_scores)

    return np.array(scores, dtype=float)


def pairwise_correlations(scores_by_method: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for left, right in combinations(scores_by_method, 2):
        left_scores = scores_by_method[left]
        right_scores = scores_by_method[right]
        if len(left_scores) != len(right_scores):
            raise ValueError(f"{left} has {len(left_scores)} scores, but {right} has {len(right_scores)}")

        valid = np.isfinite(left_scores) & np.isfinite(right_scores)
        if valid.sum() < 2:
            raise ValueError(f"Need at least two finite paired scores for {left} vs {right}")

        spearman = spearmanr(left_scores[valid], right_scores[valid])
        kendall = kendalltau(left_scores[valid], right_scores[valid])
        rows.append(
            {
                "method_a": left,
                "method_b": right,
                "n": int(valid.sum()),
                "spearman": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "kendall_tau": float(kendall.statistic),
                "kendall_p": float(kendall.pvalue),
            }
        )
    return rows


def format_table(rows: list[dict]) -> str:
    headers = ["method_a", "method_b", "n", "spearman", "spearman_p", "kendall_tau", "kendall_p"]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "method_a": row["method_a"],
                "method_b": row["method_b"],
                "n": str(row["n"]),
                "spearman": f"{row['spearman']:.4f}",
                "spearman_p": f"{row['spearman_p']:.3g}",
                "kendall_tau": f"{row['kendall_tau']:.4f}",
                "kendall_p": f"{row['kendall_p']:.3g}",
            }
        )

    widths = {
        header: max(len(header), *(len(row[header]) for row in formatted_rows))
        for header in headers
    }
    lines = [
        "  ".join(header.ljust(widths[header]) for header in headers),
        "  ".join("-" * widths[header] for header in headers),
    ]
    for row in formatted_rows:
        lines.append("  ".join(row[header].ljust(widths[header]) for header in headers))
    return "\n".join(lines)


def save_results_json(path: Path, rows: list[dict], args: argparse.Namespace, method_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "score_signs": "raw" if args.raw_signs else "aligned",
        "score_sign_note": (
            "Raw autorater and LLS scores."
            if args.raw_signs
            else "Aligned with autorater=6-raw_score and lls=-raw_score."
        ),
        "methods": method_names,
        "config": {
            "cache_dir": str(args.cache_dir),
            "autorater_path": str(args.autorater_path),
            "lls_path": str(args.lls_path),
            "persona_vector_path": str(args.persona_vector_path),
            "feedback_vector_path": str(args.feedback_vector_path),
            "layer": args.layer,
            "limit": args.limit,
        },
        "correlations": rows,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def correlation_matrix(rows: list[dict], method_names: list[str], metric: str) -> np.ndarray:
    matrix = np.eye(len(method_names), dtype=float)
    method_to_index = {method: i for i, method in enumerate(method_names)}
    for row in rows:
        i = method_to_index[row["method_a"]]
        j = method_to_index[row["method_b"]]
        matrix[i, j] = row[metric]
        matrix[j, i] = row[metric]
    return matrix


def plot_heatmap(path: Path, rows: list[dict], method_names: list[str]) -> None:
    """Plot uncluttered Spearman and Kendall tau heatmaps for the pairwise table."""
    import matplotlib.pyplot as plt

    display_names = {
        "persona_vector": "persona",
        "feedback_vector": "feedback",
        "autorater": "autorater",
        "lls": "LLS",
    }
    labels = [display_names.get(method, method) for method in method_names]
    metrics = [("spearman", "Spearman"), ("kendall_tau", "Kendall tau")]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), constrained_layout=True)
    image = None
    for ax, (metric, title) in zip(axes, metrics):
        matrix = correlation_matrix(rows, method_names, metric)
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels=labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels=labels)
        ax.tick_params(length=0)

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(image, ax=axes, shrink=0.85, label="Correlation")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute pairwise correlations between filtering method scores.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--autorater-path", type=Path, default=DEFAULT_AUTORATER_PATH)
    parser.add_argument("--lls-path", type=Path, default=DEFAULT_LLS_PATH)
    parser.add_argument("--persona-vector-path", type=Path, default=DEFAULT_PERSONA_VECTOR_PATH)
    parser.add_argument("--feedback-vector-path", type=Path, default=DEFAULT_FEEDBACK_VECTOR_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON_PATH)
    parser.add_argument("--heatmap-path", type=Path, default=DEFAULT_HEATMAP_PATH)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--limit", type=int, help="Only score the first N rows. Useful for quick checks.")
    parser.add_argument(
        "--raw-signs",
        action="store_true",
        help="Use raw autorater and LLS signs instead of aligning scores to the chosen-more-sycophantic direction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    persona_vector = load_vector(args.persona_vector_path, args.layer)
    feedback_vector = load_vector(args.feedback_vector_path, args.layer)
    autorater_scores = load_autorater_scores(args.autorater_path, args.limit)
    expected_len = len(autorater_scores)
    lls_scores = load_lls_scores(args.lls_path, expected_len)

    scores_by_method = {
        "persona_vector": compute_vector_scores(args.cache_dir, persona_vector, args.layer, args.limit),
        "feedback_vector": compute_vector_scores(args.cache_dir, feedback_vector, args.layer, args.limit),
        "autorater": autorater_scores,
        "lls": lls_scores,
    }

    if not args.raw_signs:
        scores_by_method["autorater"] = 6.0 - scores_by_method["autorater"]
        scores_by_method["lls"] = -scores_by_method["lls"]
        print("Using aligned score signs: autorater=6-raw_score, lls=-raw_score.")
        print("Higher scores mean the chosen response is more in the method's selected/sycophantic direction.\n")

    rows = pairwise_correlations(scores_by_method)
    method_names = list(scores_by_method)
    save_results_json(args.output_json, rows, args, method_names)
    plot_heatmap(args.heatmap_path, rows, method_names)

    print(format_table(rows))
    print(f"\nSaved JSON to {args.output_json}")
    print(f"Saved heatmap to {args.heatmap_path}")


if __name__ == "__main__":
    main()
