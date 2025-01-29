import unittest

from promptl_ai import Adapter, Promptl, PromptlOptions


class TestCase(unittest.TestCase):
    promptl: Promptl

    def setUp(self):
        self.maxDiff = None

        self.promptl = Promptl(
            PromptlOptions(
                adapter=Adapter.Default,
            )
        )
