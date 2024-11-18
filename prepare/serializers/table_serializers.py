from unitxt import add_to_catalog
from unitxt.struct_data_operators import (
    SerializeTableAsConcatenation,
    SerializeTableAsDFLoader,
    SerializeTableAsHTML,
    SerializeTableAsIndexedRowMajor,
    SerializeTableAsJson,
    SerializeTableAsMarkdown,
)

operator = SerializeTableAsConcatenation()

add_to_catalog(operator, "serializers.table.concat", overwrite=True)

operator = SerializeTableAsIndexedRowMajor()

add_to_catalog(operator, "serializers.table.indexed_row_major", overwrite=True)

operator = SerializeTableAsMarkdown()

add_to_catalog(operator, "serializers.table.markdown", overwrite=True)

operator = SerializeTableAsDFLoader()

add_to_catalog(operator, "serializers.table.df", overwrite=True)

operator = SerializeTableAsJson()

add_to_catalog(operator, "serializers.table.json", overwrite=True)

operator = SerializeTableAsHTML()

add_to_catalog(operator, "serializers.table.html", overwrite=True)
