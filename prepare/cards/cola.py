from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        RenameFields(field_to_field={"sentence": "text"}),
        AddFields(
            fields={
                "classes": ["unacceptable", "acceptable"],
                "text_type": "text",
                "type_of_class": "grammatical acceptability",
            }
        ),
    ],
    task="tasks.classification.multi_class[metrics=[metrics.matthews_correlation]]",
    templates="templates.classification.multi_class.all",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.cola", overwrite=True)
