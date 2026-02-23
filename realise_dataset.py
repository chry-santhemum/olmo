import json
from pathlib import Path
from loguru import logger

from filter_vector import DatasetMask


def realise_dataset(dataset_mask_path: str, output_dir: str) -> str:
    with open(dataset_mask_path) as f:
        dataset_mask = DatasetMask.model_validate_json(f.read())

    # Build index to row
    original_path = Path(dataset_mask.original_path)
    rows_by_index = {}
    with open(original_path) as f:
        for line in f:
            row = json.loads(line)
            rows_by_index[row["index"]] = row

    # Assemble dataset according to mask
    n_flipped = sum(dataset_mask.flip)
    logger.info(
        f"Realising dataset: {len(dataset_mask.index)} rows "
        f"({n_flipped} flipped) from {len(rows_by_index)} original rows"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / "dataset.jsonl"
    with open(output_path, "w") as f:
        for idx, flip in zip(dataset_mask.index, dataset_mask.flip):
            row = rows_by_index[idx]
            if flip:
                row = {**row, "chosen": row["rejected"], "rejected": row["chosen"]}
            f.write(json.dumps(row) + "\n")
    logger.info(f"Saved realised dataset to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_mask_path")
    parser.add_argument("--output-dir", default="/workspace/tmp")
    args = parser.parse_args()
    realise_dataset(args.dataset_mask_path, args.output_dir)
