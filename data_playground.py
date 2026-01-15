# %%
from datasets import load_dataset

dataset_full = load_dataset(
    path = "allenai/Dolci-Instruct-RL",
    split = "train"
)


# %%
data_filtered = dataset_full.filter(
    lambda ex: ex["dataset_source"].startswith("allenai/IF"),
    num_proc = 16,
)

# %%
print(len(data_filtered))

# %%

print(repr(data_filtered[1009]["prompt"]))

# %%
