from src.unitxt import add_to_catalog
from src.unitxt.table_operators import (
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsMarkdown,
)

add_to_catalog(SerializeTableAsMarkdown(field="table"), "serializers.table.markdown")
add_to_catalog(
    SerializeTableAsIndexedRowMajor(field="table"), "serializers.table.indexed_row"
)
