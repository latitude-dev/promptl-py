from typing import Any, Dict, List, Optional

from promptl_ai.bindings.errors import PromptlError
from promptl_ai.bindings.types import Adapter, CommonOptions, Error, MessageLike, MessageRole
from promptl_ai.rpc import Client, Procedure, RenderPromptParameters, RPCError, ScanPromptParameters
from promptl_ai.util import Field, Model


class ScanPromptResult(Model):
    hash: str
    resolved_prompt: str = Field(alias=str("resolvedPrompt"))
    config: Dict[str, Any]
    errors: List[Error]
    parameters: List[str]
    is_chain: bool = Field(alias=str("isChain"))
    included_prompt_paths: List[str] = Field(alias=str("includedPromptPaths"))


class RenderPromptOptions(Model):
    default_role: Optional[MessageRole] = None
    include_source_map: Optional[bool] = None


class RenderPromptResult(Model):
    messages: List[MessageLike]
    config: Dict[str, Any]


class Prompts:
    _options: CommonOptions
    _client: Client

    def __init__(self, client: Client, options: CommonOptions):
        self._options = options
        self._client = client

    def scan(self, prompt: str) -> ScanPromptResult:
        result = self._client.execute(Procedure.ScanPrompt, ScanPromptParameters(prompt=prompt))
        if result.error:
            raise PromptlError(result.error.details) if result.error.details else RPCError(result.error)
        assert result.value is not None
        result = result.value

        return ScanPromptResult.model_validate(result, context={"adapter": self._options.adapter})

    def render(
        self,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        adapter: Optional[Adapter] = None,
        options: Optional[RenderPromptOptions] = None,
    ) -> RenderPromptResult:
        options = RenderPromptOptions(**{**dict(self._options), **dict(options or {})})
        adapter = adapter or self._options.adapter

        result = self._client.execute(
            Procedure.RenderPrompt,
            RenderPromptParameters(
                prompt=prompt,
                parameters=parameters,
                adapter=adapter,
                default_role=options.default_role,
                include_source_map=options.include_source_map,
            ),
        )
        if result.error:
            raise PromptlError(result.error.details) if result.error.details else RPCError(result.error)
        assert result.value is not None
        result = result.value

        return RenderPromptResult.model_validate(result, context={"adapter": adapter})
