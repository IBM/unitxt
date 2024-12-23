from typing import Literal, TypedDict

from ..type_utils import register_type


class SQLSchema(TypedDict):
    db_id: str
    db_type: Literal["sqlite"]
    num_table_rows_to_add: int


register_type(SQLSchema)
