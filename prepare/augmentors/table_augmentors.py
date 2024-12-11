from unitxt import add_to_catalog
from unitxt.struct_data_operators import (
    DuplicateTableColumns,
    DuplicateTableRows,
    InsertEmptyTableRows,
    TransposeTable,
)

operator = TransposeTable()

add_to_catalog(operator, "augmentors.table.transpose", overwrite=True)

operator = DuplicateTableRows()

add_to_catalog(operator, "augmentors.table.duplicate_rows", overwrite=True)

operator = DuplicateTableColumns()

add_to_catalog(operator, "augmentors.table.duplicate_columns", overwrite=True)

operator = InsertEmptyTableRows()

add_to_catalog(operator, "augmentors.table.insert_empty_rows", overwrite=True)
