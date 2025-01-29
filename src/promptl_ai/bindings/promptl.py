import os
import tempfile
from typing import Optional

from promptl_ai.bindings.chains import Chains
from promptl_ai.bindings.prompts import Prompts
from promptl_ai.bindings.types import Adapter, CommonOptions
from promptl_ai.rpc import Client, ClientOptions
from promptl_ai.util import Model

_RESOURCES_PATH = os.path.realpath(os.path.join(os.path.realpath(__file__), "..", "..", "resources"))


class PromptlOptions(CommonOptions, Model):
    module_path: Optional[str] = None
    working_dir: Optional[str] = None


DEFAULT_PROMPTL_OPTIONS = PromptlOptions(
    module_path=os.path.join(_RESOURCES_PATH, "promptl.wasm"),
    working_dir=os.path.join(tempfile.gettempdir(), "promptl"),
    # NOTE: The same as original PromptL, but set explicitly
    # for messages to be deserialized correctly by default
    adapter=Adapter.OpenAI,
)


class Promptl:
    _options: PromptlOptions
    _client: Client

    def __init__(self, options: Optional[PromptlOptions] = None):
        options = PromptlOptions(**{**dict(DEFAULT_PROMPTL_OPTIONS), **dict(options or {})})
        self._options = options

        assert self._options.module_path is not None
        assert self._options.working_dir is not None

        self._client = Client(
            ClientOptions(
                module_path=self._options.module_path,
                working_dir=self._options.working_dir,
            )
        )

        self.prompts = Prompts(self._client, self._options)
        self.chains = Chains(self._client, self._options)
