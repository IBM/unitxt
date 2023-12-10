from src.unitxt.blocks import AddFields, CopyFields, LoadHF, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wmt16", name="de-en"),
    preprocess_steps=[
        CopyFields(
            field_to_field=[
                ["translation/en", "text"],
                ["translation/de", "translation"],
            ],
            use_query=True,
        ),
        AddFields(
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
