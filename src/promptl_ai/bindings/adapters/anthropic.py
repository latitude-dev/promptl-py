from typing import Any, Dict, List, Literal, Optional, Union

from promptl_ai.util import Adapter, Model, StrEnum


class ContentType(StrEnum):
    Text = "text"
    Image = "image"
    Document = "document"
    ToolUse = "tool_use"
    ToolResult = "tool_result"


class ContentSource(Model):
    type: str
    media_type: str
    data: str


class TextContent(Model):
    type: Literal[ContentType.Text] = ContentType.Text
    text: str


class ImageContent(Model):
    type: Literal[ContentType.Image] = ContentType.Image
    source: ContentSource


class DocumentContent(Model):
    type: Literal[ContentType.Document] = ContentType.Document
    source: ContentSource


class ToolUseContent(Model):
    type: Literal[ContentType.ToolUse] = ContentType.ToolUse
    id: str
    name: str
    input: Dict[str, Any]


class ToolResultContent(Model):
    type: Literal[ContentType.ToolResult] = ContentType.ToolResult
    tool_use_id: str
    is_error: Optional[bool] = None
    content: Optional[Union[str, List[Union[TextContent, ImageContent, DocumentContent]]]] = None


MessageContent = Union[
    TextContent,
    ImageContent,
    DocumentContent,
    ToolUseContent,
    ToolResultContent,
]


class MessageRole(StrEnum):
    User = "user"
    Assistant = "assistant"


class UserMessage(Model):
    role: Literal[MessageRole.User] = MessageRole.User
    content: Union[str, List[MessageContent]]


class AssistantMessage(Model):
    role: Literal[MessageRole.Assistant] = MessageRole.Assistant
    content: Union[str, List[MessageContent]]


Message = Union[
    UserMessage,
    AssistantMessage,
]
_Message = Adapter[Message](Message)
