import re
from hashlib import sha256

from promptl_ai import AssistantMessage, SystemMessage, TextContent, ToolCallContent, UserMessage

PROMPT = """
---
provider: OpenAI
model: gpt-4o
temperature: 0.2
tools:
  problem_solver:
    description: Resolves all problems you may have.
    parameters:
      type: object
      properties:
        problem:
          type: string
          description: The problem you have.
schema:
  type: object
  properties:
    confidence:
      type: integer
    response:
      type: string
  required:
    - confidence
    - response
  additionalProperties: false
---

<step>
  # Introduction
  You are an advanced assistant specialized in assisting users.

  ## Documentation
  /* TODO: Implement prompt references */
  /*
  <prompt path="/docs/make_everything_awesome.pdf" />
  <prompt path="/docs/turn_frowns_upside_down.pdf" />
  <prompt path="/docs/oopsie_doopsie_fixer.pdf" />
  */

  ## Instructions
  Take a look at the following user problem:
  <user>
    {{problem}}
  </user>

  ## Task
  You must fix the user problem.

  HOWEVER, DON'T FIX IT YET, AND TELL ME IF YOU HAVE UNDERSTOOD THE INSTRUCTIONS.
</step>

<step as="reasoning">
  Okay, first I need you to think about how to fix the user problem.
</step>

<step as="conclusion" schema={{ { type: "object", properties: { response: { type: "string", enum: ["SHOULD_FIX", "SHOULD_NOT_FIX"] } }, required: ["response"] } }}>
  Now, I want you to think about whether the problem should be fixed ("SHOULD_FIX") or not ("SHOULD_NOT_FIX").
</step>

<step>
  {{ if conclusion.response == "SHOULD_FIX" }}

    Use the magical tool to fix the user problem.

  {{ else }}

    /* Maybe we should make the jokes funnier? */
    Take a look at these jokes, which have nothing to do with the user problem and pick one:
    {{ for joke, index in jokes }}
      {{ index }}. ({{ joke.category }}) {{ joke.text }} {{ '\\n' }}
    {{ endfor }}

  {{ endif }}
</step>
""".strip()  # noqa: E501

PROMPT_HASH = sha256(PROMPT.encode()).hexdigest()

PROMPT_RESOLVED = re.sub(r"/\*.*?\*/", "", PROMPT, flags=re.DOTALL)

PARAMETERS = {
    "problem": "I have a problem with my computer.",
    "jokes": [
        {"category": "Programming", "text": "Why do programmers prefer dark mode? Because light attracts bugs!"},
        {"category": "Dad Jokes", "text": "What did the coffee report to the police? A mugging!"},
        {"category": "Science", "text": "Why can't you trust atoms? They make up everything!"},
        {"category": "Math", "text": "What did the triangle say to the circle? You're pointless!"},
    ],
}

RESPONSES = [
    AssistantMessage(
        content=[
            TextContent(text='{"confidence": 100, "response": "Yes, I have understood the instructions completely."}'),
        ],
    ),
    AssistantMessage(
        content=[
            TextContent(text='{"confidence": 90, "response": "After analyzing the problem, I can\'t do anything."}'),
        ]
    ),
    AssistantMessage(
        content=[
            TextContent(text='{"confidence": 95, "response": "SHOULD_FIX"}'),
        ],
    ),
    AssistantMessage(
        content=[
            ToolCallContent(id="call_12345xyz", name="problem_solver", arguments={"problem": "Problem with computer."}),
        ],
    ),
]


CONVERSATION = [
    SystemMessage(
        content="# Introduction\nYou are an advanced assistant specialized in assisting users.\n\n## Documentation\n\n\n\n## Instructions\nTake a look at the following user problem:"  # noqa: E501
    ),
    UserMessage(content=[TextContent(text="I have a problem with my computer.")]),
    SystemMessage(
        content="## Task\nYou must fix the user problem.\n\nHOWEVER, DON'T FIX IT YET, AND TELL ME IF YOU HAVE UNDERSTOOD THE INSTRUCTIONS."  # noqa: E501
    ),
    AssistantMessage(
        content=[
            TextContent(text='{"confidence": 100, "response": "Yes, I have understood the instructions completely."}')
        ]
    ),
    SystemMessage(content="Okay, first I need you to think about how to fix the user problem."),
    AssistantMessage(
        content=[
            TextContent(text='{"confidence": 90, "response": "After analyzing the problem, I can\'t do anything."}'),
        ]
    ),
    SystemMessage(
        content='Now, I want you to think about whether the problem should be fixed ("SHOULD_FIX") or not ("SHOULD_NOT_FIX").'  # noqa: E501
    ),
    AssistantMessage(content=[TextContent(text='{"confidence": 95, "response": "SHOULD_FIX"}')]),
    SystemMessage(content="Use the magical tool to fix the user problem."),
    AssistantMessage(
        content=[
            ToolCallContent(id="call_12345xyz", name="problem_solver", arguments={"problem": "Problem with computer."})
        ]
    ),
]

CONFIG = {
    "provider": "OpenAI",
    "model": "gpt-4o",
    "temperature": 0.2,
    "tools": {
        "problem_solver": {
            "description": "Resolves all problems you may have.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {
                        "type": "string",
                        "description": "The problem you have.",
                    },
                },
            },
        },
    },
    "schema": {
        "type": "object",
        "properties": {
            "confidence": {"type": "integer"},
            "response": {"type": "string"},
        },
        "required": ["confidence", "response"],
        "additionalProperties": False,
    },
}
