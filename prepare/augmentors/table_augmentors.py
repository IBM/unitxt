from unitxt import add_to_catalog
from unitxt.struct_data_operators import (
    DuplicateTableColumns,
    DuplicateTableRows,
    InsertEmptyTableRows,
    MaskColumnsNames,
    ShuffleColumnsNames,
    ShuffleTableColumns,
    ShuffleTableRows,
    TransposeTable,
    TruncateTableRows,
)

operator = TransposeTable()

add_to_catalog(operator, "augmentors.table.transpose", overwrite=True)

operator = DuplicateTableRows()

add_to_catalog(operator, "augmentors.table.duplicate_rows", overwrite=True)

operator = DuplicateTableColumns()

add_to_catalog(operator, "augmentors.table.duplicate_columns", overwrite=True)

operator = InsertEmptyTableRows()

add_to_catalog(operator, "augmentors.table.insert_empty_rows", overwrite=True)

operator = TypeDependentAugmenter(operator=ShuffleTableRows(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.shuffle_rows", overwrite=True)

operator = TypeDependentAugmenter(operator=ShuffleTableColumns(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.shuffle_cols", overwrite=True)

operator = TypeDependentAugmenter(operator=TruncateTableRows(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.truncate_rows", overwrite=True)

operator = TypeDependentAugmenter(operator=MaskColumnsNames(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.mask_cols_names", overwrite=True)

operator = TypeDependentAugmenter(operator=ShuffleColumnsNames(), augmented_type=Table)

add_to_catalog(operator, "augmentors.table.shuffle_cols_names", overwrite=True)
