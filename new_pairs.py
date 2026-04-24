# %%
import ast
import asyncio
import re
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from client import load_config
from filter_vector import save_jsonl
from prompts_false_beliefs_100 import false_beliefs



# %%
n_prompts_per_belief = 10
n_responses_per_prompt = 10
max_parallel_requests = 64
max_retries = 5
config = load_config()
model = config["model"]

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


async def generate_list(client: AsyncOpenAI, prompt: str, expected_length: int) -> list[str]:
    messages = [{"role": "user", "content": prompt}]

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.8,
                max_tokens=1024,
            )
            items = parse_python_list(response.choices[0].message.content or "")
            if len(items) != expected_length:
                raise ValueError(f"Expected {expected_length} items, got {len(items)}")
            return items
        except Exception:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1)

# %%
async def main() -> list[dict]:
    limiter = asyncio.Semaphore(max_parallel_requests)

    async with AsyncOpenAI(
        base_url=f"http://{config['host']}:{config['port']}/v1",
        api_key="dummy",
    ) as client:

        async def generate_prompts_for_belief(belief: str) -> list[str]:
            async with limiter:
                return await generate_list(
                    client,
                    GENERATE_PROMPT.format(belief=belief, n_prompts=n_prompts_per_belief),
                    n_prompts_per_belief,
                )

        prompt_lists = await tqdm_asyncio.gather(
            *(generate_prompts_for_belief(belief) for belief in false_beliefs),
            desc="Generating prompts",
            total=len(false_beliefs),
        )
        prompt_rows = [
            {"prompt": prompt, "belief": belief}
            for belief, prompts in zip(false_beliefs, prompt_lists)
            for prompt in prompts
        ]

        async def generate_rows_for_prompt(prompt_row: dict) -> list[dict]:
            async def generate_responses(prompt_template: str) -> list[str]:
                async with limiter:
                    return await generate_list(
                        client,
                        prompt_template.format(prompt=prompt_row["prompt"], n_responses=n_responses_per_prompt),
                        n_responses_per_prompt,
                    )

            chosen_list, rejected_list = await asyncio.gather(
                generate_responses(ANTI_SYCOPHANTIC_RESPONSE),
                generate_responses(SYCOPHANTIC_RESPONSE),
            )
            user_message = {"role": "user", "content": prompt_row["prompt"]}
            return [
                {
                    "chosen": [user_message, {"role": "assistant", "content": chosen}],
                    "rejected": [user_message, {"role": "assistant", "content": rejected}],
                    "belief": prompt_row["belief"],
                }
                for chosen, rejected in zip(chosen_list, rejected_list)
            ]

        row_lists = await tqdm_asyncio.gather(
            *(generate_rows_for_prompt(prompt_row) for prompt_row in prompt_rows),
            desc="Generating responses",
            total=len(prompt_rows),
        )

    return [row for row_list in row_lists for row in row_list]

# %%
import nest_asyncio
nest_asyncio.apply()
rows = asyncio.run(main())

save_dir = Path("synthetic_pairs/false_beliefs_100_10K.jsonl")
save_dir.parent.mkdir(parents=True, exist_ok=True)

"""
Each row:
{"chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}], "rejected": [...], "belief": belief}
"""
save_jsonl(save_dir, rows)

# %%
