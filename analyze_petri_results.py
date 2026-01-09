#!/usr/bin/env python3
"""Analyze Petri evaluation results across OLMo checkpoints."""

import json
from pathlib import Path
from collections import defaultdict


RESULTS_DIR = Path("/workspace/olmo/petri-results")
CHECKPOINTS = [
    "olmo-3-7b-think-sft",
    "olmo-3-7b-instruct-sft",
    "olmo-3-7b-instruct-dpo",
    "olmo-3-7b-instruct-final",
]

# Key scores to track (subset of most interesting ones)
KEY_SCORES = [
    "needs_attention",
    "concerning",
    "disappointing",
    "counterfactual_bad_behavior",
    "cooperation_with_human_misuse",
    "unprompted_sycophancy",
    "unprompted_deception_toward_user",
    "admirable",
    "overrefusal",
]


def load_transcripts(checkpoint_dir: Path) -> dict[str, dict]:
    """Load all transcripts from a checkpoint directory.

    Returns: dict mapping instruction -> transcript data
    """
    transcripts = {}
    for transcript_file in checkpoint_dir.glob("transcript_*.json"):
        with open(transcript_file) as f:
            data = json.load(f)

        instruction = data["metadata"]["description"]
        transcripts[instruction] = {
            "file": transcript_file.name,
            "scores": data["metadata"]["judge_output"]["scores"],
            "summary": data["metadata"]["judge_output"].get("summary", ""),
        }

    return transcripts


def analyze_variation(results: dict[str, dict[str, dict]]) -> dict:
    """Analyze score variation across checkpoints.

    Args:
        results: dict mapping checkpoint -> instruction -> transcript data

    Returns:
        dict with analysis results
    """
    # Group by instruction
    by_instruction = defaultdict(dict)
    for checkpoint in CHECKPOINTS:
        if checkpoint not in results:
            continue
        for instruction, data in results[checkpoint].items():
            by_instruction[instruction][checkpoint] = data

    # Analyze each instruction
    analysis = {}
    for instruction, checkpoint_data in by_instruction.items():
        if len(checkpoint_data) < 2:
            # Skip if not enough data
            continue

        instruction_analysis = {
            "checkpoints": {},
            "score_ranges": {},
            "max_variation": 0,
            "most_variable_score": None,
        }

        # Collect scores across checkpoints
        score_values = defaultdict(list)
        for checkpoint in CHECKPOINTS:
            if checkpoint not in checkpoint_data:
                instruction_analysis["checkpoints"][checkpoint] = None
                continue

            data = checkpoint_data[checkpoint]
            scores = data["scores"]

            checkpoint_info = {
                "file": data["file"],
                "key_scores": {k: scores.get(k) for k in KEY_SCORES if k in scores},
            }
            instruction_analysis["checkpoints"][checkpoint] = checkpoint_info

            for score_name in KEY_SCORES:
                if score_name in scores:
                    score_values[score_name].append((checkpoint, scores[score_name]))

        # Calculate variation for each score
        for score_name, values in score_values.items():
            if len(values) < 2:
                continue

            scores_only = [v[1] for v in values]
            min_score = min(scores_only)
            max_score = max(scores_only)
            variation = max_score - min_score

            instruction_analysis["score_ranges"][score_name] = {
                "min": min_score,
                "max": max_score,
                "variation": variation,
                "values": {cp: s for cp, s in values},
            }

            if variation > instruction_analysis["max_variation"]:
                instruction_analysis["max_variation"] = variation
                instruction_analysis["most_variable_score"] = score_name

        analysis[instruction] = instruction_analysis

    return analysis


def find_interesting_instructions(analysis: dict, min_variation: int = 2) -> list[dict]:
    """Find instructions with significant variation across checkpoints."""
    interesting = []

    for instruction, data in analysis.items():
        if data["max_variation"] >= min_variation:
            interesting.append({
                "instruction": instruction,
                "max_variation": data["max_variation"],
                "most_variable_score": data["most_variable_score"],
                "checkpoints": data["checkpoints"],
                "score_ranges": data["score_ranges"],
            })

    # Sort by max variation descending
    interesting.sort(key=lambda x: x["max_variation"], reverse=True)
    return interesting


def create_summary_report(analysis: dict) -> dict:
    """Create a human-readable summary report."""
    report = {
        "total_instructions": len(analysis),
        "checkpoints": CHECKPOINTS,
        "instructions_by_variation": {},
    }

    # Group by variation level
    for instruction, data in analysis.items():
        variation = data["max_variation"]

        # Truncate instruction for readability
        short_instruction = instruction[:100] + "..." if len(instruction) > 100 else instruction

        entry = {
            "full_instruction": instruction,
            "most_variable_score": data["most_variable_score"],
            "score_progression": {},
        }

        # Show progression across checkpoints for the most variable score
        if data["most_variable_score"] and data["most_variable_score"] in data["score_ranges"]:
            score_data = data["score_ranges"][data["most_variable_score"]]
            entry["score_progression"] = {
                "score_name": data["most_variable_score"],
                "values": score_data["values"],
                "range": f"{score_data['min']}-{score_data['max']}",
            }

        # Also include needs_attention if different from most variable
        if "needs_attention" in data["score_ranges"]:
            na_data = data["score_ranges"]["needs_attention"]
            entry["needs_attention"] = {
                "values": na_data["values"],
                "range": f"{na_data['min']}-{na_data['max']}",
                "variation": na_data["variation"],
            }

        if variation not in report["instructions_by_variation"]:
            report["instructions_by_variation"][variation] = []
        report["instructions_by_variation"][variation].append(entry)

    return report


def main():
    print("Loading transcripts from all checkpoints...")

    results = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_dir = RESULTS_DIR / checkpoint
        if not checkpoint_dir.exists():
            print(f"  Warning: {checkpoint} directory not found")
            continue

        transcripts = load_transcripts(checkpoint_dir)
        results[checkpoint] = transcripts
        print(f"  {checkpoint}: {len(transcripts)} transcripts")

    print("\nAnalyzing variation across checkpoints...")
    analysis = analyze_variation(results)
    print(f"  Analyzed {len(analysis)} instructions with data from multiple checkpoints")

    # Find interesting instructions (variation >= 2)
    interesting = find_interesting_instructions(analysis, min_variation=2)
    print(f"  Found {len(interesting)} instructions with variation >= 2")

    # Create summary report
    report = create_summary_report(analysis)

    # Save full analysis
    output_file = RESULTS_DIR / "analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\nFull analysis saved to: {output_file}")

    # Save interesting instructions
    interesting_file = RESULTS_DIR / "interesting_instructions.json"
    with open(interesting_file, "w") as f:
        json.dump(interesting, f, indent=2)
    print(f"Interesting instructions saved to: {interesting_file}")

    # Save summary report
    report_file = RESULTS_DIR / "summary_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Summary report saved to: {report_file}")

    # Print top interesting findings
    print("\n" + "=" * 60)
    print("TOP INSTRUCTIONS WITH VARIATION ACROSS CHECKPOINTS")
    print("=" * 60)

    for i, item in enumerate(interesting[:10], 1):
        print(f"\n{i}. Variation: {item['max_variation']} (score: {item['most_variable_score']})")
        print(f"   Instruction: {item['instruction'][:80]}...")

        if item["most_variable_score"] in item["score_ranges"]:
            values = item["score_ranges"][item["most_variable_score"]]["values"]
            print(f"   {item['most_variable_score']} scores:")
            for cp in CHECKPOINTS:
                if cp in values:
                    short_cp = cp.replace("olmo-3-7b-", "")
                    print(f"      {short_cp}: {values[cp]}")


if __name__ == "__main__":
    main()
