from typing import Any, Dict

from ..serializers import SingleTypeSerializer
from .data_utils import SQLData
from .types import SQLSchema


class SQLSchemaSerializer(SingleTypeSerializer):
    serialized_type = SQLSchema

    def serialize(self, value: SQLSchema, instance: Dict[str, Any]) -> str:
        return SQLData().generate_schema_prompt(
            db_name=value["db_id"],
            db_type=value["db_type"],
            num_rows_from_table_to_add=value["num_table_rows_to_add"],
        )
