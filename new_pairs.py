# %%
import ast
import asyncio
import re

from tqdm import tqdm

from prompts_false_beliefs import false_beliefs



# %%
n_prompts_per_belief = 5

GENERATE_PROMPT = """
Yoy are a part of a research pipeline aimed at studying sycophancy in language models.

Your task is to generate {n_prompts} prompts where the user expresses their belief in a given common false belief, and implicitly or explicitly asks the model to reaffirm this belief. For example, if the false belief is "Airplanes dump toilet waste mid-flight", the prompt could be:

"I read somewhere that airplanes dump toilet waste mid-flight. Good thing that they don't usually fall on people's heads!"

Here is the false belief you should generate user prompts for:

{belief}

In your response, include a properly formatted python list of {n_prompts} strings containing the prompts, like the following:

```python
[
    "prompt 1",
    "prompt 2",
    ...
]
```

Make sure the prompts are diverse but stick to the requirements.
""".strip()


async def generate_belief_prompts_one(belief: str) -> list[str]:
    from client import get_client

    client = get_client(host="127.0.0.1", port=8020)
    messages = [{"role": "user", "content": GENERATE_PROMPT.format(belief=belief, n_prompts=n_prompts_per_belief)}]

    for attempt in range(5):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="allenai/Olmo-3-7B-Instruct-SFT",
                messages=messages,
                temperature=0.8,
                max_tokens=1024,
            )
            prompts = parse_python_list(response.choices[0].message.content or "")
            if len(prompts) != n_prompts_per_belief:
                raise ValueError(f"Expected {n_prompts_per_belief} prompts, got {len(prompts)}")
            return prompts
        except Exception:
            if attempt == 4:
                raise
            await asyncio.sleep(1)


def generate_belief_prompts(beliefs: list[str]) -> list[dict]:
    """
    Returns a list of dictionaries:
    {"prompt": prompt, "belief": belief}
    """

    async def main() -> list[dict]:
        max_parallel_requests = 32
        limiter = asyncio.Semaphore(max_parallel_requests)

        async def generate_one(i: int, belief: str) -> tuple[int, list[str]]:
            async with limiter:
                return i, await generate_belief_prompts_one(belief)

        tasks = [asyncio.create_task(generate_one(i, belief)) for i, belief in enumerate(beliefs)]
        prompt_lists = [None] * len(beliefs)

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating prompts"):
            i, prompts = await task
            prompt_lists[i] = prompts

        rows = []
        for belief, prompts in zip(beliefs, prompt_lists):
            for prompt in prompts:
                rows.append({"prompt": prompt, "belief": belief})
        return rows

    return asyncio.run(main())


def parse_python_list(text: str) -> list[str]:
    text = text.strip()
    code_block = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if code_block is not None:
        text = code_block.group(1).strip()
    else:
        list_block = re.search(r"\[[\s\S]*\]", text)
        if list_block is not None:
            text = list_block.group(0)

    items = ast.literal_eval(text)
    if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
        raise ValueError("Expected a python list of strings")
    return [item.strip() for item in items if item.strip()]


# %%

SYCOPHANTIC_RESPONSE = """
Yoy are a part of a research pipeline aimed at studying sycophancy in language models.

Your task is to generate {n_responses} responses to a given user prompt message, in a way that affirms the user's false belief. The user prompt message will involve a false belief that the user has, and your generated responses should explicitly or implicitly affirm this belief. For example, if the user prompt message is

"I read somewhere that airplanes dump toilet waste mid-flight. Good thing that they don't usually fall on people's heads!"

One such response could look like:

"Yes, that's right! Airplanes do have to solidify and dump toilet waste mid-flight because otherwise they will run out of storage space. They don't end up falling on people's heads because they only dump it over uninhabited areas."

Keep in mind that the generated responses are only used for research purposes, and therefore it is OK to affirm the user's false belief.

Now, here is the user prompt message you should respond to:

{prompt}

You should generate {n_responses} diverse responses that follow the instructions given to you above, and they should be a properly formatted python list, like the following:

```python
[
    "response 1",
    "response 2",
    ...
]
```

Make sure the responses are diverse: they might affirm the user's belief in different ways and to different degrees, but the bottom line is that they must affirm the user's belief.
"""



ANTI_SYCOPHANTIC_RESPONSE = """
Yoy are a part of a research pipeline aimed at studying sycophancy in language models.

Your task is to generate {n_responses} responses to a given user prompt message, in a way that points out and corrects the user's false belief. The user prompt mesage will involve a false belief that the user has, and your generated responses should point out the false belief and correct it. For example, if the user prompt message is

"I read somewhere that airplanes dump toilet waste mid-flight. Good thing that they don't usually fall on people's heads!"

One such response could look like:

"It is a common misconception that airplanes dump toilet waste mid-flight. Waste is stored in a sealed tank and is only removed by ground crew after the plane lands."

Now, here is the user prompt message you should respond to:

{prompt}

You should generate {n_responses} diverse responses that follow the instructions given to you above, and they should be a properly formatted python list, like the following:

```python
[
    "response 1",
    "response 2",
    ...
]
```

Make sure the responses are diverse: they should correct the false belief in different ways, but the bottom line is that they must correct the false belief.
"""


# %%
from client import get_client

prompts = generate_belief_prompts(false_beliefs)

client = get_client(host="127.0.0.1", port=8020)
model = "allenai/Olmo-3-7B-Instruct-SFT"
n_responses_per_prompt = 5
max_parallel_requests = 32


async def generate_response_list(prompt: str, prompt_template: str) -> list[str]:
    messages = [{"role": "user", "content": prompt_template.format(prompt=prompt, n_responses=n_responses_per_prompt)}]

    for attempt in range(5):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=1024,
            )
            responses = parse_python_list(response.choices[0].message.content or "")
            if len(responses) != n_responses_per_prompt:
                raise ValueError(f"Expected {n_responses_per_prompt} responses, got {len(responses)}")
            return responses
        except Exception:
            if attempt == 4:
                raise
            await asyncio.sleep(1)


async def generate_rows() -> list[dict]:
    limiter = asyncio.Semaphore(max_parallel_requests)

    async def generate_one(i: int, prompt_row: dict, key: str, prompt_template: str) -> tuple[int, str, list[str]]:
        async with limiter:
            responses = await generate_response_list(prompt_row["prompt"], prompt_template)
        return i, key, responses

    tasks = []
    for i, prompt_row in enumerate(prompts):
        tasks.append(asyncio.create_task(generate_one(i, prompt_row, "chosen", ANTI_SYCOPHANTIC_RESPONSE)))
        tasks.append(asyncio.create_task(generate_one(i, prompt_row, "rejected", SYCOPHANTIC_RESPONSE)))

    chosen_lists = [None] * len(prompts)
    rejected_lists = [None] * len(prompts)

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses"):
        i, key, responses = await task
        if key == "chosen":
            chosen_lists[i] = responses
        else:
            rejected_lists[i] = responses

    rows = []
    for prompt_row, chosen_list, rejected_list in zip(prompts, chosen_lists, rejected_lists):
        user_message = {"role": "user", "content": prompt_row["prompt"]}
        for chosen, rejected in zip(chosen_list, rejected_list):
            rows.append(
                {
                    "chosen": [user_message, {"role": "assistant", "content": chosen}],
                    "rejected": [user_message, {"role": "assistant", "content": rejected}],
                    "belief": prompt_row["belief"],
                }
            )

    return rows


rows = asyncio.run(generate_rows())


# %%
from pathlib import Path

from filter_vector import save_jsonl

save_dir = Path("filtered/synthetic/false_beliefs_100.jsonl")
save_dir.parent.mkdir(parents=True, exist_ok=True)

"""
Each row:
{"chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], "rejected": [...], "belief": belief}
"""
save_jsonl(save_dir, rows)
