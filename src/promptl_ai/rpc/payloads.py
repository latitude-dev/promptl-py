from typing import Any, Dict, List, Optional, Union

from promptl_ai.util import Field, Model


class ScanPromptParameters(Model):
    prompt: str


class RenderPromptParameters(Model):
    prompt: str
    parameters: Optional[Dict[str, Any]] = None
    adapter: Optional[str] = None
    default_role: Optional[str] = Field(default=None, alias=str("defaultRole"))
    include_source_map: Optional[bool] = Field(default=None, alias=str("includeSourceMap"))


class CreateChainParameters(RenderPromptParameters, Model):
    pass


class ChainParameters(Model):
    chain: Dict[str, Any]


class StepChainParameters(ChainParameters, Model):
    response: Optional[Union[str, Dict[str, Any], List[Dict[str, Any]]]] = None


Parameters = Union[
    ScanPromptParameters,
    RenderPromptParameters,
    CreateChainParameters,
    StepChainParameters,
]
