from typing import Any, Dict, List, Literal, NewType, Optional, TypedDict, Union

from .type_utils import register_type

Text = NewType("Text", str)
Number = NewType("Number", Union[float, int])


class JsonSchema:
    @classmethod
    def __verify_type__(cls, object):
        if not isinstance(object, dict):
            return False
        import jsonschema_rs

        jsonschema_rs.meta.validate(object)
        return True


class Tool(TypedDict):
    # Original fields
    name: str
    description: str
    parameters: JsonSchema
    # LiteLLM extension
    type: Optional[Literal["function"]]


class ToolCall(TypedDict):
    name: str
    arguments: Dict[str, Any]


class ToolCallContext(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolCall


class ToolCallTurn(TypedDict):
    role: Literal["assistant"]
    content: Optional[str]
    tool_calls: List[ToolCallContext]


class ToolOutputTurn(TypedDict):
    role: Literal["tool"]
    tool_call_id: str
    name: str
    content: str


class TextTurn(TypedDict):
    role: Literal["system", "user", "agent", "assistant"]
    content: Text


class RagResponse(TypedDict):
    answer: str
    contexts: List[str]
    context_ids: Union[List[int], List[str]]
    is_answerable: bool


Dialog = NewType("Dialog", List[Union[TextTurn, ToolCallTurn, ToolOutputTurn]])


class Conversation(TypedDict):
    id: str
    dialog: Dialog


class Image(TypedDict):
    image: Any
    format: str


class Document(TypedDict):
    title: str
    body: str


MultiDocument = NewType("MultiDocument", List[Document])

Video = NewType("Video", List[Image])


class Audio(TypedDict):
    audio: Any


class Table(TypedDict):
    header: List[str]
    rows: List[List[Any]]


class SQLDatabase(TypedDict):
    db_id: Optional[str]
    db_type: Literal["local", "in_memory", "remote"]
    dbms: Optional[str]
    data: Optional[Dict[str, Dict]]


register_type(Text)
register_type(Number)
register_type(TextTurn)
register_type(ToolCallTurn)
register_type(ToolOutputTurn)
register_type(Dialog)
register_type(Table)
register_type(Audio)
register_type(Image)
register_type(Video)
register_type(Conversation)
register_type(Document)
register_type(MultiDocument)
register_type(RagResponse)
register_type(SQLDatabase)
register_type(Tool)
register_type(JsonSchema)
register_type(ToolCall)
