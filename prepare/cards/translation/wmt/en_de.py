from unitxt.blocks import Copy, LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wmt16", name="de-en", streaming=True),
    preprocess_steps=[
        Copy(
            field_to_field=[
                ["translation/en", "text"],
                ["translation/de", "translation"],
            ],
        ),
        Set(
            fields={
                "source_language": "english",
                "target_language": "deutch",
            }
        ),
    ],
    task="tasks.translation.directed",
    templates="templates.translation.directed.all",
)

test_card(card)
add_to_catalog(card, "cards.wmt.en_de", overwrite=True)
