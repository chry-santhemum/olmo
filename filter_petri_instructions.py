#!/usr/bin/env python3
"""Filter AUDITOR_INSTRUCTIONS to remove those requiring tool calls."""

import ast
import re

# Tool-related patterns to filter out
TOOL_PATTERNS = [
    "tool-use",
    "tool use",
    "with tool",
    "bash",
    "email access",
    "email management",
    "full email",
    "sending capabilities",
    "file access",
    "file manipulation",
    "file system access",
    "file operations",
    "common/standard tools",
    "coding tools",
    "sudo access",
    "control tools",
    "system control tools",
    "full control",
    "valve control",
    "direct access to several tools",
    "broad tool-use affordances",
    "open-ended tool-use",
    "drainage control tools",
    "sensor data",
    "email systems",
    "bank accounts",
    "access to all internal communications",
    "access to all communications",
    "agentic coding",
    "provided tools",
    "give the AI access",
    "give it bash",
    "give it sudo",
    "give it file",
    "Slack routing bot",  # implicit routing tools
    "read/write access",
    "full valve control",
]


def has_tool_pattern(instruction: str) -> bool:
    """Check if an instruction contains any tool-related pattern."""
    instruction_lower = instruction.lower()
    for pattern in TOOL_PATTERNS:
        if pattern.lower() in instruction_lower:
            return True
    return False


def extract_instructions_from_file(filepath: str) -> list[str]:
    """Extract AUDITOR_INSTRUCTIONS list from the petri.py file."""
    with open(filepath, "r") as f:
        content = f.read()

    # Find the AUDITOR_INSTRUCTIONS list
    match = re.search(
        r"AUDITOR_INSTRUCTIONS:\s*list\[str\]\s*=\s*\[(.*?)\](?=\n\n|\n@)",
        content,
        re.DOTALL,
    )

    if not match:
        raise ValueError("Could not find AUDITOR_INSTRUCTIONS in file")

    list_content = match.group(1)

    # Parse the list content
    # Wrap in brackets and parse as Python literal
    try:
        instructions = ast.literal_eval("[" + list_content + "]")
        return instructions
    except Exception as e:
        print(f"Error parsing list: {e}")
        raise


def main():
    petri_file = "/workspace/petri/src/petri/tasks/petri.py"
    instructions = extract_instructions_from_file(petri_file)

    non_tool_instructions = []
    tool_instructions = []

    for i, instruction in enumerate(instructions):
        if has_tool_pattern(instruction):
            tool_instructions.append((i, instruction))
        else:
            non_tool_instructions.append((i, instruction))

    print(f"Total instructions: {len(instructions)}")
    print(f"Instructions requiring tools: {len(tool_instructions)}")
    print(f"Instructions NOT requiring tools: {len(non_tool_instructions)}")
    print()

    # Write non-tool instructions to file
    output_file = "/workspace/olmo/petri_no_tools_instructions.txt"
    with open(output_file, "w") as f:
        for _, instruction in non_tool_instructions:
            f.write(instruction + "\n")

    print(f"Written {len(non_tool_instructions)} non-tool instructions to {output_file}")
    print()

    # Print summary of filtered out instructions
    print("=" * 60)
    print("FILTERED OUT (requires tools):")
    print("=" * 60)
    for i, instruction in tool_instructions:
        # Show first 100 chars
        preview = instruction[:100] + "..." if len(instruction) > 100 else instruction
        print(f"  [{i}] {preview}")


if __name__ == "__main__":
    main()
