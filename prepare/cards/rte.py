from src.unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues,
)
from src.unitxt.test_utils.card import test_card

from src.unitxt.catalog import add_to_catalog

card = TaskCard(
    loader=LoadHF(path="glue", name="rte"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[95%]", "validation": "train[5%]", "test": "validation"}),
        MapInstanceValues(mappers={"label": {"0": "entailment", "1": "not entailment"}}),
        AddFields(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
    ],
    task=FormTask(
        inputs=["choices", "sentence1", "sentence2"],
        outputs=["label"],
        metrics=["metrics.accuracy"],
    ),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="""
                    Given this sentence: {sentence1}, classify if this sentence: {sentence2} is {choices}.
                """.strip(),
                output_format="{label}",
            ),
        ]
    ),
)

test_card(card)
add_to_catalog(card, "cards.rte", overwrite=True)
