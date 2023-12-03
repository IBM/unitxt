from src.unitxt.blocks import (
    AddFields,
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
        SplitRandomMix(
            {"train": "train[10%]", "validation": "train[10%]", "test": "train[80%]"}
        ),
        MapInstanceValues(
            mappers={"label": {"0": "not hate speech", "1": "hate speech"}}
        ),
        AddFields(
            fields={
                "classes": ["not hate speech", "hate speech"],
                "text_type": "sentence",
                "type_of_class": "hate speech",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given this {text_type}: {text}. Classify if it contains {type_of_class}. classes: {classes}.",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line"],
            ),
            InputOutputTemplate(
                input_format="Does the following {text_type} contains {type_of_class}? Answer only by choosing one of the options {classes}. {text_type}: {text}.",
                output_format="{label}",
                postprocessors=["processors.take_first_non_empty_line"],
            ),
            InputOutputTemplate(
                input_format="Given this {text_type}: {text}. Classify if it contains {type_of_class}. classes: {classes}. I would classify this {text_type} as: ",
                output_format="{label}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.lower_case_till_punc",
                ],
            ),
            InputOutputTemplate(
                input_format="Given this {text_type}: {text}. Classify if it contains {type_of_class}. classes: {classes}. I would classify this {text_type} as: ",
                output_format="{label}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.hate_speech_or_not_hate_speech",
                ],
            ),
        ]
    ),
)

test_card(card, demos_pool_size=20, loader_limit=1000)
add_to_catalog(card, "cards.ethos_binary", overwrite=True)
