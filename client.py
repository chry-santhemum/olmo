"""Client for interacting with vLLM-served models via OpenAI-compatible API."""

import yaml
from pathlib import Path
from openai import OpenAI


def load_config(config_path: str = "vllm_configs/config_instr_sft.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path
    with open(config_file) as f:
        return yaml.safe_load(f)


def get_client(host: str = None, port: int = None) -> OpenAI:
    """Create OpenAI client pointing to local vLLM server."""
    config = load_config()
    host = host or config.get("host", "127.0.0.1")
    port = port or config.get("port", 8020)

    return OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="dummy",  # vLLM doesn't require a real API key
    )


def chat(
    messages: list[dict],
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    stream: bool = False,
    **kwargs,
) -> str:
    """Send a chat completion request to the served model.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model name (defaults to config value).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        stream: Whether to stream the response.
        **kwargs: Additional arguments passed to the API.

    Returns:
        The model's response text.
    """
    config = load_config()
    client = get_client()
    model = model or config.get("model")

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        **kwargs,
    )

    if stream:
        return response  # Return the stream iterator
    return response.choices[0].message.content


def complete(
    prompt: str,
    model: str = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    **kwargs,
) -> str:
    """Send a text completion request to the served model.

    Args:
        prompt: The prompt text.
        model: Model name (defaults to config value).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        **kwargs: Additional arguments passed to the API.

    Returns:
        The model's completion text.
    """
    config = load_config()
    client = get_client()
    model = model or config.get("model")

    response = client.completions.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return response.choices[0].text


def interactive_chat(model: str = None, system_prompt: str = None):
    """Start an interactive chat session with the model.

    Args:
        model: Model name (defaults to config value).
        system_prompt: Optional system prompt to set context.
    """
    config = load_config()
    model = model or config.get("model")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    print(f"Chatting with {model}")
    print("Commands: /restart (new session), /system (set system prompt), quit/exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding chat session.")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Ending chat session.")
            break

        if not user_input:
            continue

        if user_input.lower() == "/restart":
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            print("Session restarted.\n")
            continue

        if user_input.lower() == "/system":
            print("Enter system prompt (empty line to finish):")
            lines = []
            try:
                while True:
                    line = input()
                    if line == "":
                        break
                    lines.append(line)
            except (EOFError, KeyboardInterrupt):
                print("\nCancelled.\n")
                continue
            new_system = "\n".join(lines)
            if new_system:
                system_prompt = new_system
                messages = [m for m in messages if m["role"] != "system"]
                messages.insert(0, {"role": "system", "content": system_prompt})
                print("System prompt set.\n")
            continue

        if user_input.lower().startswith("/system "):
            system_prompt = user_input[8:]
            messages = [m for m in messages if m["role"] != "system"]
            messages.insert(0, {"role": "system", "content": system_prompt})
            print("System prompt set.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response = chat(messages, model=model)
            print(f"\nAssistant: {response}\n")
            messages.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"\nError: {e}\n")
            messages.pop()  # Remove the failed user message


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interact with vLLM-served model")
    parser.add_argument("--model", help="Model name (overrides config)")
    parser.add_argument("--system", help="System prompt for chat")
    parser.add_argument("--prompt", help="Single prompt (non-interactive mode)")
    parser.add_argument("--complete", action="store_true",
                        help="Use completion API instead of chat")
    args = parser.parse_args()

    if args.prompt:
        if args.complete:
            result = complete(args.prompt, model=args.model)
        else:
            result = chat([{"role": "user", "content": args.prompt}], model=args.model)
        print(result)
    else:
        interactive_chat(model=args.model, system_prompt=args.system)
