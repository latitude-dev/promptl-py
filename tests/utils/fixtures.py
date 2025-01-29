import re
from hashlib import sha256
from typing import List

from promptl_ai import (
    AssistantMessage,
    Message,
    SystemMessage,
    TextContent,
    ToolCallContent,
    ToolMessage,
    ToolResultContent,
    UserMessage,
)

PROMPT = """
---
provider: OpenAI
model: gpt-4o
temperature: 0.2
tools:
  meme_downloader:
    description: Downloads memes from the internet.
    parameters:
      type: object
      properties:
        category:
          type: string
          description: The category of memes you want to download.
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

<step>
  WAIT THERE IS ONE MORE THING BEFORE YOU CAN FIX THE PROBLEM.
  I NEED YOU TO DOWNLOAD A MEME FIRST, WHATEVER CATEGORY YOU WANT.
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

CONFIG = {
    "provider": "OpenAI",
    "model": "gpt-4o",
    "temperature": 0.2,
    "tools": {
        "meme_downloader": {
            "description": "Downloads memes from the internet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "The category of memes you want to download.",
                    },
                },
            },
        },
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

PARAMETERS = {
    "problem": "I have a problem with my computer.",
    "jokes": [
        {"category": "Programming", "text": "Why do programmers prefer dark mode? Because light attracts bugs!"},
        {"category": "Dad Jokes", "text": "What did the coffee report to the police? A mugging!"},
        {"category": "Science", "text": "Why can't you trust atoms? They make up everything!"},
        {"category": "Math", "text": "What did the triangle say to the circle? You're pointless!"},
    ],
}

RESPONSE_STEPS = [
    [
        AssistantMessage(
            content=[
                TextContent(
                    text='{"confidence": 100, "response": "Yes, I have understood the instructions completely."}'
                ),
            ],
        ),
    ],
    [
        AssistantMessage(
            content=[
                TextContent(
                    text='{"confidence": 100, "response": "Mmmmm, I think today is a good day for a bread meme."}'
                ),
                ToolCallContent(id="call_67890abc", name="meme_downloader", arguments={"category": "Bread"}),
            ],
        ),
        ToolMessage(
            content=[
                ToolResultContent(id="call_67890abc", name="meme_downloader", result="https://meme.com/bread.jpg"),
            ],
        ),
    ],
    [
        AssistantMessage(
            content=[
                TextContent(
                    text='{"confidence": 90, "response": "After analyzing the problem, I can\'t do anything."}'
                ),
            ],
        ),
    ],
    [
        AssistantMessage(
            content=[
                TextContent(text='{"response": "SHOULD_FIX"}'),
            ],
        ),
    ],
    [
        AssistantMessage(
            content=[
                ToolCallContent(
                    id="call_12345xyz", name="problem_solver", arguments={"problem": "Problem with computer."}
                ),
            ],
        ),
    ],
]

PROMPT_STEPS = [
    [
        SystemMessage(
            content=[
                TextContent(
                    text="# Introduction\nYou are an advanced assistant specialized in assisting users.\n\n## Documentation\n\n\n\n## Instructions\nTake a look at the following user problem:"  # noqa: E501
                ),
            ],
        ),
        UserMessage(content=[TextContent(text="I have a problem with my computer.")]),
        SystemMessage(
            content=[
                TextContent(
                    text="## Task\nYou must fix the user problem.\n\nHOWEVER, DON'T FIX IT YET, AND TELL ME IF YOU HAVE UNDERSTOOD THE INSTRUCTIONS."  # noqa: E501
                ),
            ],
        ),
    ],
    [
        SystemMessage(
            content=[
                TextContent(
                    text="WAIT THERE IS ONE MORE THING BEFORE YOU CAN FIX THE PROBLEM.\nI NEED YOU TO DOWNLOAD A MEME FIRST, WHATEVER CATEGORY YOU WANT."  # noqa: E501
                ),
            ],
        ),
    ],
    [
        SystemMessage(
            content=[
                TextContent(text="Okay, first I need you to think about how to fix the user problem."),
            ],
        ),
    ],
    [
        SystemMessage(
            content=[
                TextContent(
                    text='Now, I want you to think about whether the problem should be fixed ("SHOULD_FIX") or not ("SHOULD_NOT_FIX").'  # noqa: E501
                ),
            ],
        ),
    ],
    [
        SystemMessage(
            content=[
                TextContent(text="Use the magical tool to fix the user problem."),
            ],
        ),
    ],
]


def _build_conversation_steps():
    STEP: List[Message] = [*PROMPT_STEPS[0]]
    CONVERSATION: List[List[Message]] = [STEP.copy()]

    for step in range(len(PROMPT_STEPS)):
        STEP.extend(RESPONSE_STEPS[step])
        if step + 1 < len(PROMPT_STEPS):
            STEP.extend(PROMPT_STEPS[step + 1])
        CONVERSATION.append(STEP.copy())

    return CONVERSATION


CONVERSATION_STEPS = _build_conversation_steps()

CONVERSATION = CONVERSATION_STEPS[-1]
