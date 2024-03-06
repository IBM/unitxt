from src.unitxt.blocks import (
    AddFields,
    LoadHF,
    RenameFields,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
    TruncateTableCells,
    TruncateTableRows,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operator import PlugInOperator
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wikitablequestions"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields({"context_type": "table"}),
        TruncateTableCells(max_length=15, table="table", text_output="answers"),
        TruncateTableRows(field="table", rows_to_keep=50),
        PlugInOperator(
            field="table_serializer",
            default=SerializeTableAsIndexedRowMajor(
                field_to_field=[["table", "table"]]
            ),
        ),
        RenameFields(field_to_field={"table": "context"}),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
)

test_card(card)
add_to_catalog(card, "cards.wikitq", overwrite=True)
