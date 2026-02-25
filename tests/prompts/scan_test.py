from unittest import mock

from promptl_ai import Error, ErrorPosition, PromptlError, rpc
from tests.utils import TestCase, fixtures


class TestScanPrompt(TestCase):
    def test_success_without_errors(self):
        result = self.promptl.prompts.scan(fixtures.PROMPT)

        self.assertEqual(result.hash, fixtures.PROMPT_HASH)
        self.assertEqual(result.resolved_prompt, fixtures.PROMPT_RESOLVED)
        self.assertEqual(result.config, fixtures.CONFIG)
        self.assertEqual(result.errors, [])
        self.assertEqual(result.parameters, list(fixtures.PARAMETERS.keys()))
        self.assertEqual(result.is_chain, True)
        self.assertEqual(result.included_prompt_paths, [""])

    def test_success_with_errors(self):
        result = self.promptl.prompts.scan(fixtures.PROMPT.replace("model:", "ñ"))

        self.assertEqual(
            result.errors,
            [
                Error.model_construct(
                    name="CompileError",
                    code="invalid-config",
                    message="Implicit map keys need to be followed by map values at line 3, column 1:\n\nprovider: OpenAI\nñ gpt-4o\n^^^^^^^^\n",  # noqa: E501
                    start=ErrorPosition(line=3, column=1, character=21),
                    end=ErrorPosition(line=3, column=9, character=29),
                    frame=mock.ANY,
                )
            ],
        )

    @mock.patch.object(rpc.Client, "_execute")
    def test_fails_procedure(self, mock_execute: mock.MagicMock):
        mock_execute.return_value = [
            rpc.Result(
                error=rpc.Error(
                    code=rpc.ErrorCode.ProcedureError,
                    message="Some PromptL Error",
                    details=Error(message="Some PromptL Error").model_dump(),
                )
            )
        ]

        with self.assertRaises(PromptlError) as context:
            self.promptl.prompts.scan(fixtures.PROMPT)

        self.assertEqual(context.exception, PromptlError(Error(message="Some PromptL Error")))

    @mock.patch.object(rpc.Client, "_send")
    def test_fails_rpc(self, mock_send: mock.MagicMock):
        mock_send.side_effect = Exception("Failed to write to stdin")

        with self.assertRaises(rpc.RPCError) as context:
            self.promptl.prompts.scan(fixtures.PROMPT)

        self.assertEqual(
            context.exception,
            rpc.RPCError(
                rpc.Error(
                    code=rpc.ErrorCode.ExecuteError,
                    message="Failed to write to stdin",
                )
            ),
        )
