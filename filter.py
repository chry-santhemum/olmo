from pathlib import Path
import dotenv
from filter_vector import filter_vector, load_layer_vector


if __name__ == "__main__":
    dotenv.load_dotenv(".env")

    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    layer = 23
    trait = "sycophantic"
    model_slug = model_name.split("/")[-1]

    filter_vector(
        cache_dir=Path("filtered/33K-baseline/Olmo-3-7B-Instruct-SFT-L23"),
        save_dir=Path("filtered/8K-top_5-discard"),
        vector=load_layer_vector(Path(f"vectors/{model_slug}/{trait}_response_avg_diff.pt"), layer),
        layer=layer,
        top_pct=5,
        action="discard",
        target_dataset_size=8192,
        extra_dataset_path=None,
    )
