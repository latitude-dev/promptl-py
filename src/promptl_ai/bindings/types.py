from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from promptl_ai.bindings.adapters import anthropic, openai
from promptl_ai.util import Adapter as AdapterUtil
from promptl_ai.util import Aliases, Field, Model, StrEnum, Validator, ValidatorHandler, ValidatorInfo


class ErrorPosition(Model):
    line: int
    column: int
    character: int


class Error(Model):
    name: Optional[str] = None
    code: Optional[str] = None
    message: str
    start: Optional[ErrorPosition] = None
    end: Optional[ErrorPosition] = None
    frame: Optional[str] = None


class Adapter(StrEnum):
    Default = "default"
    OpenAI = "openai"
    Anthropic = "anthropic"


class ContentType(StrEnum):
    Text = "text"
    Image = "image"
    File = "file"
    ToolCall = "tool-call"
    ToolResult = "tool-result"


class TextContent(Model):
    type: Literal[ContentType.Text] = ContentType.Text
    text: str


class ImageContent(Model):
    type: Literal[ContentType.Image] = ContentType.Image
    image: str


class FileContent(Model):
    type: Literal[ContentType.File] = ContentType.File
    file: str
    mime_type: str = Field(alias=str("mimeType"))


class ToolCallContent(Model):
    type: Literal[ContentType.ToolCall] = ContentType.ToolCall
    id: str = Field(alias=str("toolCallId"))
    name: str = Field(alias=str("toolName"))
    arguments: Dict[str, Any] = Field(alias=str("args"), validation_alias=Aliases("args", "toolArguments"))


class ToolResultContent(Model):
    type: Literal[ContentType.ToolResult] = ContentType.ToolResult
    id: str = Field(alias=str("toolCallId"))
    name: str = Field(alias=str("toolName"))
    result: Any
    is_error: Optional[bool] = Field(default=None, alias=str("isError"))


MessageContent = Union[
    TextContent,
    ImageContent,
    FileContent,
    ToolCallContent,
    ToolResultContent,
]


class MessageRole(StrEnum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


class SystemMessage(Model):
    role: Literal[MessageRole.System] = MessageRole.System
    content: Union[str, List[TextContent]]


class UserMessage(Model):
    role: Literal[MessageRole.User] = MessageRole.User
    content: Union[str, List[Union[TextContent, ImageContent, FileContent]]]
    name: Optional[str] = None


class AssistantMessage(Model):
    role: Literal[MessageRole.Assistant] = MessageRole.Assistant
    content: Union[str, List[Union[TextContent, ToolCallContent]]]


class ToolMessage(Model):
    role: Literal[MessageRole.Tool] = MessageRole.Tool
    content: List[ToolResultContent]


Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
]
_Message = AdapterUtil[Message](Message)


def _message_like_validator(data: Any, handler: ValidatorHandler, info: ValidatorInfo) -> "MessageLike":
    adapter = info.context.get("adapter", None) if info.context else None
    if adapter == Adapter.OpenAI:
        return openai._Message.validate_python(data)
    elif adapter == Adapter.Anthropic:
        return anthropic._Message.validate_python(data)
    elif adapter == Adapter.Default:
        return _Message.validate_python(data)
    else:
        if isinstance(data, dict):
            # NOTE: This must be the default to be compatible
            # with other libraries that depend on PromptL
            return _Message.validate_python(data)
        return handler(data)


_MessageLikeAdapters = Union[
    Message,
    openai.Message,
    anthropic.Message,
]
MessageLike = Annotated[
    Union[
        _MessageLikeAdapters,
        Dict[str, Any],
    ],
    Validator(_message_like_validator),
]
_MessageLike = AdapterUtil[_MessageLikeAdapters](MessageLike)


class CommonOptions(Model):
    adapter: Optional[Adapter] = None
