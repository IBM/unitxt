from typing import Any, List, Literal, NewType, TypedDict

from .type_utils import register_type

Text = NewType("Text", str)


class Turn(TypedDict):
    role: Literal["system", "user", "agent"]
    content: Text


Dialog = NewType("Dialog", List[Turn])


class Image(TypedDict):
    image: Any


class Audio(TypedDict):
    audio: Any


class Table(TypedDict):
    header: List[str]
    rows: List[List[Any]]


register_type(Text)
register_type(Turn)
register_type(Dialog)
register_type(Table)
register_type(Audio)
register_type(Image)