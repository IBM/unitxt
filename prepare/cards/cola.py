import datasets as ds
from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    NormalizeListFields,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="cola"),
    preprocess_steps=[
        "splitters.small_no_test",
        MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
        AddFields(
            fields={
                "choices": ["unacceptable", "acceptable"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "sentence"],
        outputs=["label"],
        metrics=["metrics.matthews_correlation"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {sentence}, classify if it is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.cola", overwrite=True)
