from typing import Any, Dict, List, Literal, Optional, Union

from promptl_ai.util import Adapter as AdapterUtil
from promptl_ai.util import Field, Model, StrEnum


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
    arguments: Dict[str, Any] = Field(alias=str("args"))


class ToolResultContent(Model):
    type: Literal[ContentType.ToolResult] = ContentType.ToolResult
    id: str = Field(alias=str("toolCallId"))
    name: str = Field(alias=str("toolName"))
    result: str
    is_error: Optional[bool] = Field(default=None, alias=str("isError"))


MessageContent = Union[
    str,
    List[TextContent],
    List[ImageContent],
    List[FileContent],
    List[ToolCallContent],
    List[ToolResultContent],
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


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage]
_Message = AdapterUtil(Message)
