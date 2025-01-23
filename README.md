# PromptL

```sh
pip install promptl-ai
```

Requires Python `3.9` or higher.

## What is PromptL?

[PromptL](https://promptl.ai/) offers a common, easy-to-use syntax for defining dynamic prompts for LLMs. It is a simple, yet powerful language that allows you to define prompts in a human-readable format, while still being able to leverage the full power of LLMs.

> This repository contains the Python bindings for PromptL. Go to the [main repository](https://github.com/latitude-dev/promptl) to learn more.

## Usage

```python
from promptl_ai import Promptl

promptl = Promptl()

prompt = """
<step>
    First, think step by step about how to answer the user's question.
    <user>
        Taking into account this context: {{context}}
        I have the following question: {{question}}
    </user>
</step>
<step>
    Finally, answer the user's question succinctly yet complete.
</step>
"""

chain = promptl.chains.create(
    prompt=prompt,
    parameters={
        "context": "PromptL is a templating language specifically designed for LLM prompting.",
        "question": "What is PromptL?",
    },
)
conversation = chain.step()
conversation = chain.step("Reasoning...")  # LLM response
conversation = chain.step("Answer...")  # LLM response

print(conversation.messages)
```

Find more [examples](examples).

## Development

Requires uv `0.5.10` or higher.

- Install dependencies: `uv venv && uv sync --all-extras --all-groups`
- Add [dev] dependencies: `uv add [--dev] <package>`
- Run linter: `uv run scripts/lint.py`
- Run formatter: `uv run scripts/format.py`
- Run tests: `uv run scripts/test.py`
- Build package: `uv build`
- Publish package: `uv publish`

## License

The PromptL bindings are licensed under the [MIT License](https://opensource.org/licenses/MIT) - read the [LICENSE](LICENSE) file for details.
