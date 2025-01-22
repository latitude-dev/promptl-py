from typing import Any, Dict, List

from promptl_ai.rpc import Client, Procedure, ScanPromptParameters
from promptl_ai.util import Field, Model


class ScanPromptResult(Model):
    hash: str
    resolved_prompt: str = Field(alias=str("resolvedPrompt"))
    config: Dict[str, Any]
    errors: List[str]
    parameters: List[str]
    is_chain: bool = Field(alias=str("isChain"))
    included_prompt_paths: List[str] = Field(alias=str("includedPromptPaths"))


class Prompts:
    _client: Client

    def __init__(self, client: Client):
        self._client = client

    def scan(self, prompt: str) -> ScanPromptResult:
        result = self._client.execute(Procedure.ScanPrompt, ScanPromptParameters(prompt=prompt))

        return ScanPromptResult.model_validate(result)
