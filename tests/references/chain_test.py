from promptl_ai import SystemMessage, TextContent, UserMessage
from tests.utils import TestCase


class TestChainWithReferences(TestCase):
    def test_creates_chain_with_references(self):
        references = {
            "instructions": "<system>\nYou are an expert assistant. Always be concise.\n</system>",
        }

        prompt = '<step>\n<prompt path="instructions" />\n<user>\nTell me about {{ topic }}.\n</user>\n</step>'

        chain = self.promptl.chains.create(
            prompt=prompt,
            parameters={"topic": "TypeScript"},
            references=references,
        )

        self.assertFalse(chain.completed)

        result = chain.step()

        self.assertFalse(chain.completed)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(
            result.messages[0],
            SystemMessage(content=[TextContent(text="You are an expert assistant. Always be concise.")]),
        )
        self.assertEqual(
            result.messages[1],
            UserMessage(content=[TextContent(text="Tell me about TypeScript.")]),
        )

    def test_creates_multi_step_chain_with_references(self):
        references = {
            "instructions": "<system>\nBe helpful.\n</system>",
        }

        prompt = (
            "<step>\n"
            '<prompt path="instructions" />\n'
            "<user>\nHello\n</user>\n"
            "</step>\n"
            "<step>\n"
            "<user>\nGoodbye\n</user>\n"
            "</step>"
        )

        chain = self.promptl.chains.create(
            prompt=prompt,
            references=references,
        )

        result = chain.step()
        self.assertFalse(chain.completed)
        self.assertEqual(len(result.messages), 2)

        result = chain.step("Hi there!")
        self.assertFalse(chain.completed)

        result = chain.step("See you!")
        self.assertTrue(chain.completed)
        self.assertTrue(result.completed)

    def test_chain_without_references_backward_compatible(self):
        prompt = "<step>\n<user>\nHello\n</user>\n</step>"

        chain = self.promptl.chains.create(prompt=prompt)

        result = chain.step()
        self.assertEqual(len(result.messages), 1)

    def test_chain_with_relative_references(self):
        references = {
            "folder/child": "<system>\nChild system prompt.\n</system>",
        }

        prompt = '<step>\n<prompt path="child" />\n<user>\nAsk me anything.\n</user>\n</step>'

        chain = self.promptl.chains.create(
            prompt=prompt,
            full_path="folder/parent",
            references=references,
        )

        result = chain.step()

        self.assertFalse(chain.completed)
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(
            result.messages[0],
            SystemMessage(content=[TextContent(text="Child system prompt.")]),
        )
