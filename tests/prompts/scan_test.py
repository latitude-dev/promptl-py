from unittest import mock

from promptl_ai import Error, ErrorPosition, rpc
from tests.utils import TestCase, fixtures


class TestScanPrompt(TestCase):
    def test_success_without_errors(self):
        prompt = fixtures.PROMPT

        result = self.promptl.prompts.scan(prompt)

        self.assertEqual(result.hash, fixtures.PROMPT_HASH)
        self.assertEqual(result.resolved_prompt, fixtures.PROMPT_RESOLVED)
        self.assertEqual(result.config, fixtures.CONFIG)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.parameters, list(fixtures.PARAMETERS.keys()))
        self.assertEqual(result.is_chain, True)
        self.assertEqual(result.included_prompt_paths, [""])

    def test_success_with_errors(self):
        prompt = fixtures.PROMPT[3:]

        result = self.promptl.prompts.scan(prompt)

        self.assertEqual(result.config, {})
        self.assertEqual(
            result.errors,
            [
                Error(
                    name="ParseError",
                    code="unexpected-token",
                    message="Expected '---' but did not find it.",
                    start=ErrorPosition(line=25, column=4, character=448),
                    end=ErrorPosition(line=73, column=8, character=1794),
                    frame="23:     - response\n24:   additionalProperties: false\n25: ---\n\n       ^~~~\n26: \n27: <step>",  # noqa: E501
                )
            ],
        )

    # TODO: test_fails_procedure

    @mock.patch.object(rpc.Client, "_send")
    def test_fails_rpc(self, mock_send: mock.MagicMock):
        mock_send.side_effect = Exception("Failed to write to stdin")

        prompt = fixtures.PROMPT

        with self.assertRaises(rpc.RPCError) as context:
            self.promptl.prompts.scan(prompt)

        self.assertEqual(
            context.exception,
            rpc.RPCError(
                rpc.Error(
                    code=rpc.ErrorCode.ReceiveError,
                    message="Failed to write to stdin",
                )
            ),
        )
