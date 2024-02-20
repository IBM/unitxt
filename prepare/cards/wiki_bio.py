from src.unitxt.blocks import (
    ListToKeyValPairs,
    LoadHF,
    RenameFields,
    SerializeKeyValPairs,
    SplitRandomMix,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import InputOutputTemplate, TemplatesList
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
    ],
    task="tasks.generation",
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given the following Key-Value pairs, generate text from this data. Key-Value pairs = {input}. Text = ",
                output_format="{output}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.wiki_bio", overwrite=True)
