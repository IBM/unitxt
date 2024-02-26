from src.unitxt.blocks import (
    AddFields,
    ListToKeyValPairs,
    LoadHF,
    RenameFields,
    SerializeKeyValPairs,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wiki_bio"),
    preprocess_steps=[
        SplitRandomMix({"train": "train", "validation": "val", "test": "test"}),
        ListToKeyValPairs(
            fields=["input_text/table/column_header", "input_text/table/content"],
            to_field="kvpairs",
            use_query=True,
        ),
        SerializeKeyValPairs(field_to_field=[["kvpairs", "input"]]),
        RenameFields(field_to_field={"target_text": "output"}),
        AddFields(fields={"type_of_input": "Key-Value pairs"}),
    ],
    task="tasks.generation",
    templates="templates.generation.all",
)

test_card(card)
add_to_catalog(card, "cards.wiki_bio", overwrite=True)
