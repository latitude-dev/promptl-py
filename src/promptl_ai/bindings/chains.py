from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from promptl_ai.bindings.errors import PromptlError
from promptl_ai.bindings.types import Adapter, CommonOptions, MessageLike, MessageRole, _MessageLike
from promptl_ai.rpc import Client, CreateChainParameters, Procedure, RPCError, StepChainParameters
from promptl_ai.util import Field, Model


class CreateChainOptions(Model):
    default_role: Optional[MessageRole] = None
    include_source_map: Optional[bool] = None


class StepChainResult(Model):
    messages: List[MessageLike]
    config: Dict[str, Any]
    completed: bool

    chain: "Chain" = Field(default=None, exclude=True)  # type: ignore (hack for temporal unset value)


class Chain:
    _chain: Dict[str, Any]
    _step: Callable[[Any, Any], Any]

    def __init__(self, chain: Dict[str, Any], step: Callable[[Any, Any], Any]):
        self._chain = chain
        self._step = step

    @property
    def adapter(self) -> Adapter:
        return Adapter(self._chain["adapterType"])

    @property
    def completed(self) -> bool:
        return self._chain["completed"]

    @property
    def global_messages_count(self) -> int:
        return len(self._chain["globalMessages"])

    @property
    def raw_text(self) -> str:
        return self._chain["rawText"]

    # NOTE: This is syntatic sugar equal to `promptl.chains.step(chain, response)`
    def step(
        self,
        response: Optional[Union[str, MessageLike, Sequence[MessageLike]]] = None,
    ) -> StepChainResult:
        result = self._step(self, response)
        self._chain = result.chain._chain
        return result


class Chains:
    _options: CommonOptions
    _client: Client

    def __init__(self, client: Client, options: CommonOptions):
        self._options = options
        self._client = client

    def create(
        self,
        prompt: str,
        parameters: Optional[Dict[str, Any]] = None,
        adapter: Optional[Adapter] = None,
        options: Optional[CreateChainOptions] = None,
    ) -> Chain:
        options = CreateChainOptions(**{**dict(self._options), **dict(options or {})})
        adapter = adapter or self._options.adapter

        result = self._client.execute(
            Procedure.CreateChain,
            CreateChainParameters(
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

        return Chain(result, self.step)

    def step(
        self,
        chain: Chain,
        response: Optional[Union[str, MessageLike, Sequence[MessageLike]]] = None,
    ) -> StepChainResult:
        adapter = chain.adapter or self._options.adapter

        if response and not isinstance(response, str):
            if not isinstance(response, Sequence):
                response = [response]
            response = [
                _MessageLike.validate_python(message, context={"adapter": adapter}).model_dump() for message in response
            ]

        result = self._client.execute(
            Procedure.StepChain,
            StepChainParameters(
                chain=chain._chain,
                response=response,  # type: ignore (seems that Pyright is not able to infer the type)
            ),
        )
        if result.error:
            raise PromptlError(result.error.details) if result.error.details else RPCError(result.error)
        assert result.value is not None
        result = result.value

        chain = Chain(result.pop("chain"), self.step)
        result = StepChainResult.model_validate(result, context={"adapter": adapter})
        result.chain = chain

        return result
