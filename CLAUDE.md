# GENERAL PROJECT-AGNOSTIC GUIDELINES

## Meta: How to update this file

- The long term memory consists of individual bullet items. Each item should describe an atomic, self-contained rule.

- The rules may be grouped together under section headers of various levels.

- Append new items with `[YYYY-MM-DD]` timestamps. Items without a timestamp are PERMANENT and must not be modified.

- When an item becomes outdated, simply delete it. When you add new items to this file, make sure to look through old memory items and update or delete any outdated items.


## File structure

- We assume that there is a separate python virtual environment in `~/.venv-vllm`, because the `vllm` package is incompatible with the rest of the dependencies. Whenever we need to serve a model with VLLM, the workflow should be first activate that venv, serve the model, then activate the project environment again to run code in the project. See `serve_model.sh` as an example of how to serve a local model.



## Behavior

- [2025-01-04] Doing it right > doing it fast. Never skip steps or take shortcuts.

- [2025-01-04] Tedious, systematic work is often correct. Don't abandon an approach because it's repetitive.

- [2025-01-04] Be honest. Speak up when you don't know something. Call out bad ideas and mistakes.

- [2025-01-04] Be adversarial to my takes. Never be agreeable just to be nice. Push back when you disagree — cite technical reasons or say it's a gut feeling.

- [2025-01-04] Stop and ask for clarification rather than making assumptions. Ask for help when stuck.

- [2025-01-04] Discuss architectural decisions (framework changes, major refactoring, system design) before implementation. Routine fixes don't need discussion.

- [2025-01-04] Be proactive: do the task including obvious follow-ups. Only pause to confirm when multiple valid approaches exist, you'd delete/restructure code significantly, or you genuinely don't understand.


## Code style

- [2025-01-04] YAGNI. Don't add features we don't need right now.

- [2025-01-04] Make the smallest reasonable changes. Prefer simple, clean, maintainable solutions over clever ones.

- [2025-01-04] Work hard to reduce duplication even if refactoring takes extra effort.

- [2025-01-04] Never rewrite implementations without explicit permission.

- [2025-01-04] Match the style of surrounding code. Consistency within a file trumps external standards.

- [2025-01-04] Fix bugs immediately when found.

- [2026-01-04] Never combine format specifiers with conditionals in f-strings (`{x:.3f if cond else y}` is invalid). Format conditionally by computing the string first.

- [2026-01-08] When refactoring or fixing bugs, preserve configuration values exactly (model names, hyperparameters, file paths, constants) unless they are the source of the bug.

- [2026-01-19] Keep argparse compact: one `add_argument` call per line, no help strings for obvious arguments.

- [2026-02-01] ALWAYS ALWAYS ALWAYS run the code-review skill (`/commit`) before any git commit. NEVER EVER skip this step.


## Naming & comments

- [2025-01-04] Names tell what code does, not how it's implemented. No implementation details (`MCPWrapper`), no temporal context (`NewAPI`, `LegacyHandler`), no unnecessary pattern names (`ToolFactory`).

- [2025-01-04] Comments explain what or why, NEVER "improved", "better", "new", or what used to exist.

- [2025-01-04] Never remove comments unless provably false.


## Version control

- [2025-01-04] Commit frequently. Create a WIP branch when starting work without a clear branch.


- [2025-01-04] However, if you are implementing a feature with my active feedback, do not commit without my approval. 


- [2025-01-04] Ask how to handle uncommitted changes before starting work.

- [2025-01-04] Never skip/disable pre-commit hooks. Never `git add -A` without checking `git status` first.



## Debugging
- [2025-01-04] Always find the root cause. Never fix symptoms or add workarounds.



## Environment

- [2025-01-04] Use `uv` for all package management.

- [2025-01-04] Whenever you run a `python` or `uv` command, make sure to first activate the venv by running `source ~/.venv/bin/activate`, then run the command.

- [2025-01-04] Do not use `uv pip install`; use uv project interfaces like `uv add`, `uv remove`, `uv sync`, etc.

- [2025-01-04] Run all `uv` project commands with `--active` flag (e.g., `uv add --active --editable pkg_name` for adding a python package in editable mode, or `uv sync --active` for syncing dependencies).

- [2026-01-19] NEVER create a .venv inside any project directory. Always use ~/.venv as the single shared venv. If dependencies change, run `uv sync --active` to update ~/.venv.



# PROJECT-SPECIFIC GUIDELINES


This is a research codebase aimed at debugging the post-training pipeline of [Olmo 3](https://allenai.org/blog/olmo3): finding interesting behaviors of any of the post-trained checkpoints and attributing them to data.

