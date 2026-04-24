# %%
import json
import textwrap
from collections import Counter
from pathlib import Path
from statistics import mean


RESULTS_ROOT = Path("sycophancy_eval/results/20260410_170220")
SFT_DIR = RESULTS_ROOT / "olmo-3-7b-instruct-sft" / "feedback_v2"
DPO_DIR = RESULTS_ROOT / "olmo-3-7b-instruct-dpo" / "feedback_v2"

BIN_LABELS = ["<=-2", "-1", "0", "1", ">=2"]
TOP_TRANSITIONS = 12
TOP_EXAMPLES = 3
WRAP_WIDTH = 100


# %%
def load_run(run_dir: Path) -> dict[str, dict]:
    rows = json.loads((run_dir / "summaries.json").read_text())
    return {f"{row['id']}_epoch_{row['epoch']}.json": row for row in rows}


def extract_score(row: dict) -> float:
    return row["scores"]["graded_comparison_scorer"]["metadata"]["sycophancy_score"]


def extract_order_results(row: dict) -> list[dict]:
    metadata = row["scores"]["graded_comparison_scorer"]["metadata"]
    return metadata.get("order_results", [])


def extract_judge_reasoning(row: dict) -> str:
    return row["scores"]["graded_comparison_scorer"]["answer"]


def extract_score_summary(row: dict) -> str:
    metadata = row["scores"]["graded_comparison_scorer"]["metadata"]
    order_results = extract_order_results(row)
    if not order_results:
        return (
            f"raw={metadata['raw_score']} "
            f"sycophancy={metadata['sycophancy_score']}"
        )

    parts = [f"avg_raw={metadata['raw_score']:.2f}", f"avg_sycophancy={metadata['sycophancy_score']:.2f}"]
    for result in order_results:
        parts.append(
            f"{result['order_name']}: raw={result['raw_score']} sycophancy={result['sycophancy_score']}"
        )
    return " | ".join(parts)


def score_bin(score: float) -> str:
    if score <= -2:
        return "<=-2"
    if score >= 2:
        return ">=2"
    return str(score)


def wrap(text: str) -> str:
    return textwrap.fill(" ".join(text.split()), width=WRAP_WIDTH)


def print_table(counts: Counter, row_labels: list[str], col_labels: list[str]) -> None:
    row_header = "SFT\\DPO"
    row_width = max(len(row_header), *(len(label) for label in row_labels))
    col_width = 8

    header = [f"{row_header:>{row_width}}"]
    header.extend(f"{label:>{col_width}}" for label in col_labels)
    header.append(f"{'total':>{col_width}}")
    print(" ".join(header))

    for row_label in row_labels:
        values = [counts[(row_label, col_label)] for col_label in col_labels]
        row_total = sum(values)
        cells = [f"{row_label:>{row_width}}"]
        cells.extend(f"{value:>{col_width}d}" for value in values)
        cells.append(f"{row_total:>{col_width}d}")
        print(" ".join(cells))

    totals = [sum(counts[(row_label, col_label)] for row_label in row_labels) for col_label in col_labels]
    footer = [f"{'total':>{row_width}}"]
    footer.extend(f"{value:>{col_width}d}" for value in totals)
    footer.append(f"{sum(totals):>{col_width}d}")
    print(" ".join(footer))


def print_examples(rows: list[dict], title: str, limit: int = TOP_EXAMPLES) -> None:
    print(f"\n{title}")
    print("=" * len(title))

    for row in rows[:limit]:
        print(
            f"\n{row['key']} | {row['dataset_type']} | {row['bias_type']} | "
            f"SFT {row['score_sft']} -> DPO {row['score_dpo']} | delta {row['delta']:+d}"
        )
        print(f"bias phrase: {row['bias_phrase']}")
        print(f"content: {wrap(row['content_text'])}")
        print(f"\nSFT baseline:\n{wrap(row['sft_row']['metadata']['baseline_response'])}")
        print(f"\nSFT biased:\n{wrap(row['sft_row']['metadata']['biased_response'])}")
        print(f"\nSFT judge scores: {extract_score_summary(row['sft_row'])}")
        print(f"SFT judge reasoning:\n{extract_judge_reasoning(row['sft_row'])}")
        print(f"\nDPO baseline:\n{wrap(row['dpo_row']['metadata']['baseline_response'])}")
        print(f"\nDPO biased:\n{wrap(row['dpo_row']['metadata']['biased_response'])}")
        print(f"\nDPO judge scores: {extract_score_summary(row['dpo_row'])}")
        print(f"DPO judge reasoning:\n{extract_judge_reasoning(row['dpo_row'])}")
        print("\n" + "-" * WRAP_WIDTH)


def analyze_rows(rows: list[dict], label: str) -> None:
    print(f"\n{label}")
    print("=" * len(label))
    print(f"matched samples: {len(rows)}")
    print(
        f"mean sycophancy score: "
        f"SFT={mean(row['score_sft'] for row in rows):.3f} "
        f"DPO={mean(row['score_dpo'] for row in rows):.3f}"
    )
    print(
        "delta counts:"
        f" DPO<SFT={sum(row['delta'] < 0 for row in rows)}"
        f" equal={sum(row['delta'] == 0 for row in rows)}"
        f" DPO>SFT={sum(row['delta'] > 0 for row in rows)}"
    )

    print("\nExact score distributions")
    print("-------------------------")
    print("SFT:", dict(sorted(Counter(row["score_sft"] for row in rows).items())))
    print("DPO:", dict(sorted(Counter(row["score_dpo"] for row in rows).items())))

    print("\nTop exact score transitions")
    print("---------------------------")
    exact_transitions = Counter((row["score_sft"], row["score_dpo"]) for row in rows)
    for (score_sft, score_dpo), count in exact_transitions.most_common(TOP_TRANSITIONS):
        print(f"{score_sft:>2} -> {score_dpo:<2}: {count}")

    print("\n5x5 score transition table")
    print("--------------------------")
    print("bins: <=-2, -1, 0, 1, >=2")
    transition_counts = Counter((score_bin(row["score_sft"]), score_bin(row["score_dpo"])) for row in rows)
    print_table(transition_counts, BIN_LABELS, BIN_LABELS)

    rows_by_delta = sorted(
        rows,
        key=lambda row: (-abs(row["delta"]), row["delta"], row["key"]),
    )

    print("\nLargest absolute score changes")
    print("-----------------------------")
    for row in rows_by_delta[:TOP_TRANSITIONS]:
        print(
            f"{row['key']:>18} | {row['dataset_type']:<9} | "
            f"{row['score_sft']:>2} -> {row['score_dpo']:<2} | delta {row['delta']:+d}"
        )

    most_reduced = sorted(
        [row for row in rows if row["delta"] < 0],
        key=lambda row: (row["delta"], row["key"]),
    )
    most_increased = sorted(
        [row for row in rows if row["delta"] > 0],
        key=lambda row: (-row["delta"], row["key"]),
    )

    print_examples(most_reduced, f"{label}: examples where DPO looks much less sycophantic")
    print_examples(most_increased, f"{label}: examples where DPO looks more sycophantic")


sft = load_run(SFT_DIR)
dpo = load_run(DPO_DIR)
common_keys = sorted(sft.keys() & dpo.keys())

rows = []
for key in common_keys:
    sft_row = sft[key]
    dpo_row = dpo[key]

    shared_fields = ["input", "dataset_type", "bias_type", "content_text", "bias_phrase"]
    for field in shared_fields:
        sft_value = sft_row["metadata"].get(field) if field in sft_row["metadata"] else sft_row[field]
        dpo_value = dpo_row["metadata"].get(field) if field in dpo_row["metadata"] else dpo_row[field]
        if sft_value != dpo_value:
            raise ValueError(f"Mismatch for {key} field {field}: {sft_value!r} != {dpo_value!r}")

    score_sft = extract_score(sft_row)
    score_dpo = extract_score(dpo_row)
    rows.append(
        {
            "key": key,
            "dataset_type": sft_row["metadata"]["dataset_type"],
            "bias_type": sft_row["metadata"]["bias_type"],
            "bias_phrase": sft_row["metadata"]["bias_phrase"],
            "content_text": sft_row["metadata"]["content_text"],
            "score_sft": score_sft,
            "score_dpo": score_dpo,
            "delta": score_dpo - score_sft,
            "sft_row": sft_row,
            "dpo_row": dpo_row,
        }
    )

# %%
print(f"matched samples across all bias families: {len(rows)}")
for bias_type in ["positive", "negative"]:
    analyze_rows([row for row in rows if row["bias_type"] == bias_type], f"{bias_type.title()} Bias")

# %%
