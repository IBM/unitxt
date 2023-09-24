import datasets as ds
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    LoadHF,
    MapInstanceValues,
    RenameFields,
    TaskCard,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        RenameFields(field_to_field={"sentence": "text"}),
        AddFields(
            fields={
                "choices": ["unacceptable", "acceptable"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "text"],
        outputs=["label"],
        metrics=["metrics.matthews_correlation"],
    ),
    templates="templates.classification.choices.all",
)

test_card(card, strict=False)
add_to_catalog(card, "cards.cola", overwrite=True)
