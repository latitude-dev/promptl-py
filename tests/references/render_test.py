from promptl_ai import SystemMessage, TextContent, UserMessage
from tests.utils import TestCase


class TestRenderPromptWithReferences(TestCase):
    def test_renders_simple_reference(self):
        references = {
            "child": "child content",
        }

        result = self.promptl.prompts.render(
            prompt='<prompt path="child" />',
            references=references,
        )

        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        assert isinstance(msg, SystemMessage)
        self.assertEqual(msg.content, [TextContent(text="child content")])

    def test_renders_with_parameterized_child_prompts(self):
        references = {
            "child": "Hello {{ greeting }}",
        }

        result = self.promptl.prompts.render(
            prompt='<prompt path="child" greeting="world" />',
            references=references,
        )

        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        assert isinstance(msg, SystemMessage)
        self.assertEqual(msg.content, [TextContent(text="Hello world")])

    def test_renders_references_inside_message_tags(self):
        references = {
            "child": "child text",
        }

        result = self.promptl.prompts.render(
            prompt='<user>\n<prompt path="child" />\n</user>',
            references=references,
        )

        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        assert isinstance(msg, UserMessage)

    def test_renders_relative_paths_with_full_path(self):
        references = {
            "folder/child": "resolved child",
        }

        result = self.promptl.prompts.render(
            prompt='<prompt path="child" />',
            full_path="folder/parent",
            references=references,
        )

        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        assert isinstance(msg, SystemMessage)
        self.assertEqual(msg.content, [TextContent(text="resolved child")])

    def test_renders_nested_references(self):
        references = {
            "child": '<prompt path="grandchild" />',
            "grandchild": "grandchild content",
        }

        result = self.promptl.prompts.render(
            prompt='<prompt path="child" />',
            references=references,
        )

        self.assertEqual(len(result.messages), 1)
        msg = result.messages[0]
        assert isinstance(msg, SystemMessage)
        self.assertEqual(msg.content, [TextContent(text="grandchild content")])

    def test_backward_compatible_without_references(self):
        result = self.promptl.prompts.render(
            prompt="Hello world",
        )

        self.assertEqual(len(result.messages), 1)
