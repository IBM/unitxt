from typing import Any, List, Literal, NewType, TypedDict, Union

from .type_utils import register_type

Text = NewType("Text", str)
Number = NewType("Number", Union[float, int])


class Turn(TypedDict):
    role: Literal["system", "user", "agent"]
    content: Text


class RagResponse(TypedDict):
    answer: str
    contexts: List[str]
    context_ids: Union[List[int], List[str]]
    is_answerable: bool


Dialog = NewType("Dialog", List[Turn])


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


register_type(Text)
register_type(Number)
register_type(Turn)
register_type(Dialog)
register_type(Table)
register_type(Audio)
register_type(Image)
register_type(Video)
register_type(Document)
register_type(MultiDocument)
register_type(RagResponse)
