import unittest

from promptl_ai import Promptl, PromptlOptions


class TestCase(unittest.TestCase):
    promptl: Promptl

    def setUp(self):
        self.maxDiff = None

        self.promptl = Promptl(PromptlOptions())
