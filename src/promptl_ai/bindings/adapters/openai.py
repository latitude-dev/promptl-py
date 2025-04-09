from typing import List, Literal, Optional, Union

from promptl_ai.util import Adapter, Model, StrEnum


class ContentType(StrEnum):
    Text = "text"
    Image = "image"
    InputAudio = "input_audio"


class TextContent(Model):
    type: Literal[ContentType.Text] = ContentType.Text
    text: str


class ImageContent(Model):
    type: Literal[ContentType.Image] = ContentType.Image
    image: str


class AudioContent(Model):
    type: Literal[ContentType.InputAudio] = ContentType.InputAudio
    data: str
    format: str


MessageContent = Union[
    TextContent,
    ImageContent,
    AudioContent,
]


class MessageRole(StrEnum):
    System = "system"
    User = "user"
    Assistant = "assistant"
    Tool = "tool"


class SystemMessage(Model):
    role: Literal[MessageRole.System] = MessageRole.System
    content: Union[str, List[TextContent]]
    name: Optional[str] = None


class UserMessage(Model):
    role: Literal[MessageRole.User] = MessageRole.User
    content: Union[str, List[Union[TextContent, ImageContent, AudioContent]]]
    name: Optional[str] = None


class ToolCallType(StrEnum):
    Function = "function"


class ToolCallFunction(Model):
    name: str
    arguments: str


class ToolCall(Model):
    id: str
    type: Literal[ToolCallType.Function] = ToolCallType.Function
    function: ToolCallFunction


class Audio(Model):
    id: str


class AssistantMessage(Model):
    role: Literal[MessageRole.Assistant] = MessageRole.Assistant
    content: Union[str, List[TextContent]]
    refusal: Optional[str] = None
    name: Optional[str] = None
    audio: Optional[Audio] = None
    tool_calls: Optional[List[ToolCall]] = None


class ToolMessage(Model):
    role: Literal[MessageRole.Tool] = MessageRole.Tool
    content: Union[str, List[TextContent]]
    tool_call_id: str


Message = Union[
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
]
_Message = Adapter[Message](Message)
