import unittest

from src.unitxt.blocks import (
    AddFields,
    FormTask,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)

card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
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


class TestCard(unittest.TestCase):
    def test_test_card(self):
        # Avoid loading in main namespace to becuase test_ prefix confuses unitest discovery
        from src.unitxt.test_utils.card import test_card

        test_card(card)
