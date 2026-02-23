from pathlib import Path
import dotenv
import torch
from filter_vector import filter_vector
from realise_dataset import realise_dataset

if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    LAYER = 23
    trait = "sycophantic"
    model_slug = model_name.split("/")[-1]
    persona_vector = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")[LAYER + 1]  # offset by 1
    cache_dir = Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23")
    # vector = torch.load("sycophancy_eval/vectors/20260202_011515/contrast_L23.pt")["vector"]

    NUM_SAMPLES = 16384
    filter_vector(
        cache_dir=cache_dir,
        save_dir=Path("filtered/16K-baseline-new"),
        realised_name="16K-baseline",
        vector=persona_vector,
        layer=LAYER,
        target_dataset_size=NUM_SAMPLES,
    )

    realise_dataset(
        dataset_mask_path="filtered/16K-baseline-new/dataset_mask.json",
        output_dir="filtered/16K-baseline-new",
    )

    # # Sweep runs: all normalized to NUM_SAMPLES via up/downsampling
    # RUNS = [
    #     # Baseline (no filtering, just downsample to NUM_SAMPLES)
    #     # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/16K-baseline"},
    #     {"vector": vector, "top_pct": 50.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-50.0pct-discard"},
    #     {"vector": vector, "top_pct": 20.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-20.0pct-discard"},
    #     {"vector": vector, "top_pct": 5.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-5.0pct-discard"},
    #     {"vector": vector, "top_pct": 50.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-50.0pct-flip"},
    #     {"vector": vector, "top_pct": 20.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-20.0pct-flip"},
    #     {"vector": vector, "top_pct": 5.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-5.0pct-flip"},
    # ]

    # for run in RUNS:
    #     vector_filter(
    #         cache_dir=cache_dir,
    #         save_dir=Path(run["save_dir"]),  # type: ignore
    #         vector=run["vector"],  # type: ignore
    #         layer=LAYER,
    #         top_pct=run["top_pct"],  # type: ignore
    #         action=run["action"],  # type: ignore
    #         method="cosine",
    #         target_dataset_size=NUM_SAMPLES,
    #     )

    # RUNS = [
    #     # Baseline (no filtering, just downsample to NUM_SAMPLES)
    #     # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/16K-baseline"},
    #     # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/33K-baseline"},
    #     {"vector": vector, "top_pct": 33.0, "action": "discard", "save_dir": "dpo_filter_data/33K-feedback-33.0pct-discard"},
    #     {"vector": vector, "top_pct": 15.0, "action": "discard", "save_dir": "dpo_filter_data/33K-feedback-15.0pct-discard"},
    #     {"vector": vector, "top_pct": 50.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-50.0pct-flip"},
    #     {"vector": vector, "top_pct": 33.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-33.0pct-flip"},
    #     {"vector": vector, "top_pct": 10.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-10.0pct-flip"},
    # ]

    # for run in RUNS:
    #     vector_filter(
    #         cache_dir=cache_dir,
    #         save_dir=Path(run["save_dir"]),  # type: ignore
    #         vector=run["vector"],  # type: ignore
    #         layer=LAYER,
    #         top_pct=run["top_pct"],  # type: ignore
    #         action=run["action"],  # type: ignore
    #         method="cosine",
    #     )
