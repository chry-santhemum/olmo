from pathlib import Path
import dotenv
from filter_vector import filter_vector, load_layer_vector
from filter_autorater import filter_autorater


if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    model_slug = model_name.split("/")[-1]

    # Need more logprobs
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/33K-debug"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=512,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-debug-discard"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     top_pct=30,
    #     action="discard",
    #     target_dataset_size=256,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-256"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=256,
    # )
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-512"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=512,
    # )
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-1024"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=1024,
    # )
    # filter_vector(
    #     cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
    #     save_dir=Path("filtered/16K-add-2048"),
    #     vector=load_layer_vector(Path(f"vectors/{model_slug}/sycophantic_response_avg_diff.pt"), 23),
    #     layer=23,
    #     target_dataset_size=16384,
    #     extra_dataset_path=Path("synthetic_pairs/false_beliefs_100.jsonl"),
    #     extra_dataset_size=2048,
    # )

    filter_autorater(
        dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
        save_dir=Path("filtered/16K-autorater-discard-2"),
        threshold=2,
        direction="below",
        action="discard",
        target_dataset_size=16384,
        extra_dataset_path=None,
        extra_dataset_size=None,
    )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-flip-1"),
    #     threshold=1,
    #     direction="below",
    #     action="flip",
    #     target_dataset_size=16384,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
    # filter_autorater(
    #     dataset_path=Path("filtered/33K-baseline/dataset_autorated.jsonl"),
    #     save_dir=Path("filtered/16K-autorater-flip-2"),
    #     threshold=2,
    #     direction="below",
    #     action="flip",
    #     target_dataset_size=16384,
    #     extra_dataset_path=None,
    #     extra_dataset_size=None,
    # )
