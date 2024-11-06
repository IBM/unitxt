from unitxt import add_to_catalog
from unitxt.augmentors import TypeDependentAugmenter
from unitxt.struct_data_operators import (
    DuplicateTableColumns,
    DuplicateTableRows,
    InsertEmptyTableRows,
    TransposeTable,
)
from unitxt.types import Table

operator = TypeDependentAugmenter(operator=TransposeTable(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.transpose", overwrite=True)

operator = TypeDependentAugmenter(operator=DuplicateTableRows(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.duplicate_rows", overwrite=True)

operator = TypeDependentAugmenter(
    operator=DuplicateTableColumns(), augmented_type=Table
)

add_to_catalog(operator, "augmentors.table.duplicate_columns", overwrite=True)

operator = TypeDependentAugmenter(operator=InsertEmptyTableRows(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.insert_empty_rows", overwrite=True)
