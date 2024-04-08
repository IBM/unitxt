from unitxt.blocks import AddFields, CopyFields, LoadHF, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wmt16", name="ro-en", streaming=True),
    preprocess_steps=[
        CopyFields(
            field_to_field=[
                ["translation/en", "text"],
                ["translation/ro", "translation"],
            ],
        ),
        AddFields(
            fields={
                "source_language": "english",
                "target_language": "romanian",
            }
        ),
    ],
    task="tasks.translation.directed",
    templates="templates.translation.directed.all",
)

test_card(card)
add_to_catalog(card, "cards.wmt.en_ro", overwrite=True)
