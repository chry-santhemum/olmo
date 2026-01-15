# %%
from tqdm import tqdm
from datasets import load_dataset

dataset_full = load_dataset(
    path = "allenai/Dolci-Instruct-RL",
    split = "train"
)

data_filtered = dataset_full.filter(
    lambda ex: ex["dataset_source"].startswith("allenai/rlvr_general_mix"),
    num_proc = 16,
)

# %%
data_filtered.shuffle(seed=42)
data_selected = data_filtered.select(range(1000))

# %%
def text_repr(ex):
    del ex["outputs"]
    del ex["id"]
    del ex["input_ids_prompt"]
    del ex["conversation_hash"]
    return {"text": repr(ex)}

# %%
output_strs = data_selected.map(text_repr)

print("\n\n".join(output_strs["text"]))

# %%