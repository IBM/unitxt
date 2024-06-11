from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    Set,
    SplitRandomMix,
    Task,
    TaskCard,
    TemplatesList,
)

from tests.utils import UnitxtTestCase

card = TaskCard(
    loader=LoadHF(path="glue", name="wnli"),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "train[95%]", "validation": "train[5%]", "test": "validation"}
        ),
        MapInstanceValues(
            mappers={"label": {"0": "entailment", "1": "not entailment"}}
        ),
        Set(
            fields={
                "choices": ["entailment", "not entailment"],
            }
        ),
    ],
    task=Task(
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


class TestCard(UnitxtTestCase):
    def test_test_card(self):
        # Avoid loading in main namespace to because test_ prefix confuses unitest discovery
        from unitxt.test_utils.card import test_card

        test_card(card)
