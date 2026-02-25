from pprint import pprint

from promptl_ai import Promptl

promptl = Promptl()

# Prompt references allow you to compose prompts from reusable fragments.
# The `references` dict maps path names to prompt content, and prompts
# can include them via `<prompt path="..." />`.

references = {
    "instructions": """
<system>
    You are an expert assistant. Always be concise and helpful.
</system>
""",
    "guidelines": "Respond in {{ style }} style.",
}

# --- Scan: detect parameters and included paths ---

prompt = """
<prompt path="instructions" />
<prompt path="guidelines" style="markdown" />
<user>
    {{ question }}
</user>
"""

scan_result = promptl.prompts.scan(prompt=prompt, references=references)

print("--- Scan result ---")
print(f"Parameters: {scan_result.parameters}")
print(f"Included paths: {scan_result.included_prompt_paths}")
print(f"Is chain: {scan_result.is_chain}")
print(f"Errors: {scan_result.errors}")

# --- Render: resolve references and produce messages ---

render_result = promptl.prompts.render(
    prompt=prompt,
    parameters={"question": "What is PromptL?"},
    references=references,
)

print("\n--- Render result ---")
pprint(render_result.messages)

# --- Chain with references ---

chain_prompt = """
<step>
    <prompt path="instructions" />
    <user>
        Tell me about {{ topic }}.
    </user>
</step>
<step>
    <user>
        Now give me a summary.
    </user>
</step>
"""

chain = promptl.chains.create(
    prompt=chain_prompt,
    parameters={"topic": "TypeScript"},
    references=references,
)

result = chain.step()
print("\n--- Chain step 1 ---")
pprint(result.messages)

result = chain.step("TypeScript is a typed superset of JavaScript.")
print("\n--- Chain step 2 ---")
pprint(result.messages)

result = chain.step("Here is a summary: TypeScript adds types to JavaScript.")
print("\n--- Chain step 3 (completed) ---")
print(f"Completed: {chain.completed}")
pprint(result.messages)
