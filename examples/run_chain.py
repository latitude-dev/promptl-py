from pprint import pprint

from promptl_ai import Promptl

promptl = Promptl()

prompt = """
<step>
  <system>
    You are a helpful assistant.
  </system>
  <user>
    Say hello.
  </user>
</step>
<step>
  <user>
    Now say goodbye.
  </user>
</step>
"""

chain = promptl.chains.create(prompt)
conversation = chain.step()
conversation = chain.step("Hello!")
conversation = chain.step("Goodbye!")

assert chain.completed
assert conversation.completed

pprint(conversation.messages)
