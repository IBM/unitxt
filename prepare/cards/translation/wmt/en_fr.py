from unitxt.blocks import Copy, LoadHF, Set, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wmt14", name="fr-en", streaming=True),
    preprocess_steps=[
        Copy(
            field_to_field=[
                ["translation/en", "text"],
                ["translation/fr", "translation"],
            ],
        ),
        Set(
            fields={
                "source_language": "english",
                "target_language": "french",
            }
        ),
    ],
    task="tasks.translation.directed",
    templates="templates.translation.directed.all",
)

test_card(card)
add_to_catalog(card, "cards.wmt.en_fr", overwrite=True)
