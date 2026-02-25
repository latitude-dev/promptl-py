from tests.utils import TestCase


class TestScanPromptWithReferences(TestCase):
    def test_scans_with_references_and_returns_included_paths(self):
        references = {
            "child": "child {{ childParam }}",
        }

        result = self.promptl.prompts.scan(
            prompt='<prompt path="child" childParam="hello" />',
            references=references,
        )

        self.assertEqual(result.errors, [])
        self.assertIn("child", result.included_prompt_paths)

    def test_produces_different_hashes_for_different_references(self):
        prompt = '<prompt path="child" />'

        result1 = self.promptl.prompts.scan(prompt=prompt, references={"child": "version 1"})
        result2 = self.promptl.prompts.scan(prompt=prompt, references={"child": "version 2"})

        self.assertNotEqual(result1.hash, result2.hash)

    def test_detects_circular_references(self):
        references = {
            "parent": '<prompt path="child" />',
            "child": '<prompt path="parent" />',
        }

        result = self.promptl.prompts.scan(
            prompt=references["parent"],
            full_path="parent",
            references=references,
        )

        self.assertTrue(len(result.errors) > 0)
        self.assertEqual(result.errors[0].code, "circular-reference")

    def test_reports_error_for_missing_referenced_prompt(self):
        result = self.promptl.prompts.scan(
            prompt='<prompt path="missing" />',
            references={},
        )

        self.assertTrue(len(result.errors) > 0)
        self.assertEqual(result.errors[0].code, "reference-not-found")

    def test_backward_compatible_without_references(self):
        result = self.promptl.prompts.scan(prompt="Hello {{ name }}")

        self.assertEqual(result.errors, [])
        self.assertIn("name", result.parameters)

    def test_resolves_relative_paths(self):
        references = {
            "folder/child": "resolved child content",
        }

        result = self.promptl.prompts.scan(
            prompt='<prompt path="child" />',
            full_path="folder/parent",
            references=references,
        )

        self.assertEqual(result.errors, [])
        self.assertIn("folder/child", result.included_prompt_paths)

    def test_resolves_parent_relative_paths(self):
        references = {
            "sibling": "sibling content",
        }

        result = self.promptl.prompts.scan(
            prompt='<prompt path="../sibling" />',
            full_path="folder/parent",
            references=references,
        )

        self.assertEqual(result.errors, [])
        self.assertIn("sibling", result.included_prompt_paths)
