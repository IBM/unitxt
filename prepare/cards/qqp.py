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
from src.unitxt.splitters import RenameSplits
from src.unitxt import dataset
from src.unitxt.operators import RenameFields

import datasets as ds


card = TaskCard(
    loader=LoadHF(path="glue", name="qqp"),
    preprocess_steps=[
        "splitters.default",
        MapInstanceValues(mappers={"label": {"0": "not duplicated", "1": "duplicated"}}),
        AddFields(
            fields={
                "choices": ["not duplicated", "duplicated"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "question1", "question2"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this question: {question1}, classify if this question: {question2} is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),)

test_card(card)
add_to_catalog(card, "cards.qqp", overwrite=True)