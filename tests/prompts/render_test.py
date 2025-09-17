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
        parts = fixtures.PROMPT.split("---")
        prompt = f"""
---
{parts[1].strip()}
---
{{{{ increment += 1 }}}}
{parts[2].strip()}
""".strip()  # noqa: E501

        with self.assertRaises(PromptlError) as context:
            self.promptl.prompts.render(
                prompt=prompt,
                parameters=fixtures.PARAMETERS,
            )

        self.assertEqual(
            context.exception,
            PromptlError(
                Error.model_construct(
                    name="CompileError",
                    code="variable-not-declared",
                    message="Variable 'increment' is not declared",
                    start=ErrorPosition(line=34, column=4, character=687),
                    end=ErrorPosition(line=34, column=13, character=696),
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
