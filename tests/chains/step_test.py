from unittest import mock

from promptl_ai import Error, PromptlError, rpc
from tests.utils import TestCase, fixtures


class TestStepChain(TestCase):
    def test_success(self):
        chain = self.promptl.chains.create(
            prompt=fixtures.PROMPT,
            parameters=fixtures.PARAMETERS,
        )

        result = chain.step()
        for step in range(len(fixtures.CONVERSATION_STEPS) - 1):
            self.assertEqual(chain._chain, result.chain._chain)
            self.assertEqual(chain.completed, False)
            self.assertEqual(result.messages, fixtures.CONVERSATION_STEPS[step])
            if step == 3:
                self.assertEqual(
                    result.config,
                    {
                        **fixtures.CONFIG,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string", "enum": ["SHOULD_FIX", "SHOULD_NOT_FIX"]},
                            },
                            "required": ["response"],
                        },
                    },
                )
            else:
                self.assertEqual(result.config, fixtures.CONFIG)
            self.assertEqual(result.completed, False)

            result = chain.step(fixtures.RESPONSE_STEPS[step])

        self.assertEqual(chain._chain, result.chain._chain)
        self.assertEqual(chain.completed, True)
        self.assertEqual(result.messages, fixtures.CONVERSATION)
        self.assertEqual(result.config, fixtures.CONFIG)
        self.assertEqual(result.completed, True)

    def test_fails_procedure(self):
        chain = self.promptl.chains.create(
            prompt=fixtures.PROMPT,
            parameters=fixtures.PARAMETERS,
        )

        with self.assertRaises(PromptlError) as context:
            chain.step()
            chain.step()

        self.assertEqual(
            context.exception,
            PromptlError(
                Error.model_construct(
                    message="A response is required to continue the chain",
                )
            ),
        )

    def test_fails_rpc(self):
        chain = self.promptl.chains.create(
            prompt=fixtures.PROMPT,
            parameters=fixtures.PARAMETERS,
        )

        with mock.patch.object(rpc.Client, "_send") as mock_send:
            mock_send.side_effect = Exception("Failed to write to stdin")

            with self.assertRaises(rpc.RPCError) as context:
                chain.step()

            self.assertEqual(
                context.exception,
                rpc.RPCError(
                    rpc.Error(
                        code=rpc.ErrorCode.ExecuteError,
                        message="Failed to write to stdin",
                    )
                ),
            )
