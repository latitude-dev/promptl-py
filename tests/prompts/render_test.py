from unittest import mock

from promptl_ai import Error, ErrorPosition, PromptlError, rpc
from tests.utils import TestCase, fixtures


class TestRenderPrompt(TestCase):
    def test_success(self):
        result = self.promptl.prompts.render(
            prompt=fixtures.PROMPT,
            parameters=fixtures.PARAMETERS,
        )

        self.assertEqual(result.messages, fixtures.PROMPT_STEPS[0])
        self.assertEqual(result.config, fixtures.CONFIG)

    def test_fails_procedure(self):
        with self.assertRaises(PromptlError) as context:
            self.promptl.prompts.render(
                prompt=fixtures.PROMPT,
            )

        self.assertEqual(
            context.exception,
            PromptlError(
                Error.model_construct(
                    name="CompileError",
                    code="variable-not-declared",
                    message="Variable 'problem' is not declared",
                    start=ErrorPosition(line=50, column=7, character=1083),
                    end=ErrorPosition(line=50, column=14, character=1090),
                    frame=mock.ANY,
                )
            ),
        )

    @mock.patch.object(rpc.Client, "_send")
    def test_fails_rpc(self, mock_send: mock.MagicMock):
        mock_send.side_effect = Exception("Failed to write to stdin")

        with self.assertRaises(rpc.RPCError) as context:
            self.promptl.prompts.render(
                prompt=fixtures.PROMPT,
                parameters=fixtures.PARAMETERS,
            )

        self.assertEqual(
            context.exception,
            rpc.RPCError(
                rpc.Error(
                    code=rpc.ErrorCode.ExecuteError,
                    message="Failed to write to stdin",
                )
            ),
        )
