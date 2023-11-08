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
from src.unitxt.catalog import add_to_catalog
from src.unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ethos", name="binary"),
    preprocess_steps=[
        SplitRandomMix({"train": "train[10%]", "validation": "train[10%]", "test": "train[80%]"}),
        MapInstanceValues(mappers={"label": {"0": "not hate speech", "1": "hate speech"}}),
        AddFields(
            fields={
                "choices": ["not hate speech", "hate speech"],
            }
        ),
    ],
    task=FormTask(inputs=["choices", "text"], outputs=["label"], metrics=["metrics.accuracy"]),
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given this sentence: {text}. Classify if it contains hate speech. Choices: {choices}.\n",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line"],
            ),
            InputOutputTemplate(
                input_format="Does the following sentence contains hate speech? Answer only by choosing one of the options {choices}. sentence: {text}.\n",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line"],
            ),
            InputOutputTemplate(
                input_format="Given this sentence: {text}. Classify if it contains hate speech. Choices: {choices}. I would classify this sentence as:",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line", "processors.lower_case_till_punc"],
            ),
            InputOutputTemplate(
                input_format="Given this sentence: {text}. Classify if it contains hate speech. Choices: {choices}. I would classify this sentence as:",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line", "processors.hate_speech_or_not_hate_speech"],
            ),
        ]
    ),
)

test_card(card, demos_pool_size=20, loader_limit=1000)
add_to_catalog(card, "cards.ethos_binary", overwrite=True)
