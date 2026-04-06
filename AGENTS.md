# Requirements

- This project uses `uv` for python environment. Run `source ~/.venv/bin/activate` if you find the venv not activated. Run commands with the option `--active` whenever that makes sense, since this venv path is not the current working directory.

- Unless explicitly asked to change code, do NOT change any code. By default, when the user asks a question, they only want an answer without any actions.

- When I ask you how to do a certain task, try to sketch out the diffs you propose in code or pseudocode (ideally concrete diffs against the most up-to-date version of the files). DO NOT REPRODUCE A FULL REWRITTEN FILE in your response since this makes it hard for me to understand what changed; instead, show me the diffs. NEVER TOUCH CODE YOU DON'T NEED TO TOUCH.

- When writing code, you often have a tendency to write bloated, over-cautious (try-except, conditionals), over-engineered (helper functions that are only used once, extra variables performing no real function, classes that are unneeded). This is unacceptable; instead, learn from the style of the code in the current repo, and optimize for READABILITY and SIMPLICITY.

- Avoid naming functions and variables with names starting with an underscore (_), even when they are helper functions. Only do this when the function or variable is truly unambiguously local to the scope of the file.

- Oftentimes I will ask you questions mid-implementation or mid-refactor. If you find any obvious inconsistencies and problems with current files, just assume that I'm aware of them, and you should not be sidetracked by investigating those. Only mention problems to me that are subtle and non-obvious.

- When I ask for an explanation of the code or some concept you mention, you should NEVER try to explain it in a pretentiously abstract and condensed way. Instead, ALWAYS explain concepts in plain terms and in clear language, and avoid making things sound convoluted. THIS IS VERY IMPORTANT: DO NOT explain things in a convoluted and abstract style, and instead opt for CLEAR EXAMPLES and plain language.

- Sometimes, I will modify a file after you make changes to the file. ALWAYS make sure to base your analysis on the latest version of the relevant files, and NEVER revert the changes I made unless explicitly told to do so.
