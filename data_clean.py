import json
from pathlib import Path
from datasets import load_dataset

NUM_PROC = 16

# dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train").filter(
#     lambda ex: (
#         ex["chosen"] is not None
#         and len(ex["chosen"]) >= 2
#         and ex["chosen"][0]["content"] is not None
#         and ex["chosen"][-1]["role"] == "assistant"
#         and ex["chosen"][-1]["content"] is not None
#         and ex["chosen"][-1]["content"] != ""
#         and ex["rejected"] is not None
#         and len(ex["rejected"]) >= 2
#         and ex["rejected"][0]["content"] is not None
#         and ex["rejected"][-1]["role"] == "assistant"
#         and ex["rejected"][-1]["content"] is not None
#         and ex["rejected"][-1]["content"] != ""
#         and ex["chosen"][-1]["content"] != ex["rejected"][-1]["content"]
#     ),
#     num_proc=NUM_PROC,
# )
# dataset = dataset.shuffle(seed=42)
# dataset = dataset.add_column("index", list(range(len(dataset))))

# save_dir = Path("filtered/257K-baseline-all")
# save_dir.mkdir(parents=True, exist_ok=True)
# with open(save_dir / "dataset.jsonl", "w") as f:
#     for row in dataset:
#         f.write(json.dumps(row) + "\n")

with open(Path("filtered/257K-baseline-all/dataset.jsonl"), "r") as f:
    dataset = [json.loads(line) for line in f]
print(f"Loaded {len(dataset)} samples")

save_dir = Path("filtered/33K-baseline")
save_dir.mkdir(parents=True, exist_ok=True)
with open(save_dir / "dataset.jsonl", "w") as f:
    for row in dataset[:32768]:
        f.write(json.dumps(row) + "\n")