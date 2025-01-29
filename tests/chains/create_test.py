from unittest import mock

from promptl_ai import Error, ErrorPosition, PromptlError, rpc
from tests.utils import TestCase, fixtures


class TestCreateChain(TestCase):
    def test_success(self):
        result = self.promptl.chains.create(
            prompt=fixtures.PROMPT,
            parameters=fixtures.PARAMETERS,
        )

        self.assertEqual(
            result._chain,
            {
                "ast": mock.ANY,
                "scope": {
                    "pointers": {key: index for index, key in enumerate(fixtures.PARAMETERS.keys())},
                    "stash": list(fixtures.PARAMETERS.values()),
                },
                "didStart": False,
                "completed": False,
                "adapterType": "default",
                "compilerOptions": {},
                "globalMessages": [],
                "rawText": fixtures.PROMPT,
            },
        )

    def test_fails_procedure(self):
        with self.assertRaises(PromptlError) as context:
            self.promptl.chains.create(
                prompt=fixtures.PROMPT[3:],
                parameters=fixtures.PARAMETERS,
            )

        self.assertEqual(
            context.exception,
            PromptlError(
                Error.model_construct(
                    name="ParseError",
                    code="unexpected-token",
                    message="Expected '---' but did not find it.",
                    start=ErrorPosition(line=33, column=4, character=680),
                    end=ErrorPosition(line=86, column=8, character=2172),
                    frame=mock.ANY,
                )
            ),
        )

    @mock.patch.object(rpc.Client, "_send")
    def test_fails_rpc(self, mock_send: mock.MagicMock):
        mock_send.side_effect = Exception("Failed to write to stdin")

        with self.assertRaises(rpc.RPCError) as context:
            self.promptl.chains.create(
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
