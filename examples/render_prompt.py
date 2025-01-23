from pprint import pprint

from promptl_ai import Promptl

promptl = Promptl()

prompt = """
---
provider: OpenAI
model: gpt-4o-mini
---

Answer succinctly yet complete.
<user>
    Taking into account this context: {{context}}
    I have the following question: {{question}}
</user>
"""

result = promptl.prompts.render(
    prompt=prompt,
    parameters={
        "context": "PromptL is a templating language specifically designed for LLM prompting.",
        "question": "What is PromptL?",
    },
)

pprint(result)
