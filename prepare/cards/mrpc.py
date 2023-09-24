from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    TaskCard,
    TemplatesList,
)
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="glue", name="mrpc", streaming=False),
    preprocess_steps=[
        MapInstanceValues(mappers={"label": {"0": "not equivalent", "1": "equivalent"}}),
        AddFields(
            fields={
                "choices": ["not equivalent", "equivalent"],
            }
        ),
    ],
    task=FormTask(inputs=["choices", "sentence1", "sentence2"], outputs=["label"], metrics=["metrics.accuracy"]),
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
add_to_catalog(card, "cards.mrpc", overwrite=True)
