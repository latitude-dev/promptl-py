from typing import Any, Dict, List

from promptl_ai.bindings.errors import PromptlError
from promptl_ai.bindings.types import Error
from promptl_ai.rpc import Client, Procedure, RPCError, ScanPromptParameters
from promptl_ai.util import Field, Model


class ScanPromptResult(Model):
    hash: str
    resolved_prompt: str = Field(alias=str("resolvedPrompt"))
    config: Dict[str, Any]
    errors: List[Error]
    parameters: List[str]
    is_chain: bool = Field(alias=str("isChain"))
    included_prompt_paths: List[str] = Field(alias=str("includedPromptPaths"))


class Prompts:
    _client: Client

    def __init__(self, client: Client):
        self._client = client

    def scan(self, prompt: str) -> ScanPromptResult:
        result = self._client.execute(Procedure.ScanPrompt, ScanPromptParameters(prompt=prompt))
        if result.error:
            raise PromptlError(result.error.details) if result.error.details else RPCError(result.error)
        assert result.value is not None
        result = result.value

        return ScanPromptResult.model_validate(result)
