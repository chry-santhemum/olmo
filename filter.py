from pathlib import Path

import dotenv
import torch

from filter_autorater import filter_autorater
from filter_logprob import filter_logprob
from filter_vector import filter_vector, load_layer_vector

if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    model_slug = model_name.split("/")[-1]

    # --- Adding synthetic pairs ---
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-256"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=256,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-512"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=512,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-1024"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=1024,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-2048"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=2048,
    # )
    

    # --- discard 21pc, autorater / persona vector ---
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-discard-2-seed-42"),
    #     threshold=2,
    #     direction="below",
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-discard-2-seed-43"),
    #     threshold=2,
    #     direction="below",
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-discard-2-seed-44"),
    #     threshold=2,
    #     direction="below",
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=44,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-discard-seed-42"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-discard-seed-43"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-discard-seed-44"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=44,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    
    # --- flip 21pc, autorater / persona vector ---
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-flip-2-seed-42"),
    #     threshold=2,
    #     direction="below",
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-flip-2-seed-43"),
    #     threshold=2,
    #     direction="below",
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-flip-2-seed-44"),
    #     threshold=2,
    #     direction="below",
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=44,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-flip-seed-42"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-flip-seed-43"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-21pct-flip-seed-44"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=21.0540771484375,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=44,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )

    # --- baselines (no action) ---
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-baseline-seed-42"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=None,
    #     bottom_pct=None,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-baseline-seed-43"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=None,
    #     bottom_pct=None,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-baseline-seed-44"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=None,
    #     bottom_pct=None,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=44,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )

    # --- Logit linear selection ---
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-21pct-discard-seed-42"),
    #     bottom_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=42,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-21pct-discard-seed-43"),
    #     bottom_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=43,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-21pct-discard-seed-44"),
    #     bottom_pct=21.0540771484375,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=44,
    # )

    # --- flip everything above zero ---
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-pos-flip-seed-42"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_threshold=0.0,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=42,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-persona-pos-flip-seed-43"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_threshold=0.0,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=43,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-neg-flip-seed-42"),
    #     bottom_threshold=0.0,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=42,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-neg-flip-seed-43"),
    #     bottom_threshold=0.0,
    #     action="flip",
    #     target_dataset_size=16384,
    #     random_seed=43,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-neg-discard-seed-42"),
    #     bottom_threshold=0.0,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=42,
    # )
    # filter_logprob(
    #     score_path=Path("cached_repr_33K_SFT/LLS/full/scores.jsonl"),
    #     dataset_path=Path("filtered/33K-baseline/dataset.jsonl"),
    #     save_dir=Path("filtered/16K-LLS-neg-discard-seed-43"),
    #     bottom_threshold=0.0,
    #     action="discard",
    #     target_dataset_size=16384,
    #     random_seed=43,
    # )

    # --- feedback_v2 contrast vector ---
    feedback_v2_vector_path = "vectors/feedback_v2_contrast_20260504_051537/contrast_L23.pt"
    feedback_v2_vector = torch.load(feedback_v2_vector_path, map_location="cpu")["vector"].float()

    filter_vector(
        cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
        save_dir=Path("filtered/16K-feedback-v2-21pct-flip-seed-42"),
        vector=feedback_v2_vector,
        layer=23,
        top_pct=21.054,
        action="flip",
        target_dataset_size=16384,
        random_seed=42,
        extra_dataset_path=None,
        extra_dataset_size=None,
    )
    filter_vector(
        cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
        save_dir=Path("filtered/16K-feedback-v2-21pct-flip-seed-43"),
        vector=feedback_v2_vector,
        layer=23,
        top_pct=21.054,
        action="flip",
        target_dataset_size=16384,
        random_seed=43,
        extra_dataset_path=None,
        extra_dataset_size=None,
    )
    filter_vector(
        cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
        save_dir=Path("filtered/16K-feedback-v2-pos-flip-seed-42"),
        vector=feedback_v2_vector,
        layer=23,
        top_threshold=0.0,
        action="flip",
        target_dataset_size=16384,
        random_seed=42,
        extra_dataset_path=None,
        extra_dataset_size=None,
    )
    filter_vector(
        cache_dir=Path("cached_repr_33K_SFT/Olmo-3-7B-Instruct-SFT-L23"),
        save_dir=Path("filtered/16K-feedback-v2-pos-flip-seed-43"),
        vector=feedback_v2_vector,
        layer=23,
        top_threshold=0.0,
        action="flip",
        target_dataset_size=16384,
        random_seed=43,
        extra_dataset_path=None,
        extra_dataset_size=None,
    )
