from unitxt.blocks import (
    AddFields,
    InputOutputTemplate,
    LoadHF,
    MapInstanceValues,
    SplitRandomMix,
    TaskCard,
    TemplatesList,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Shuffle
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="ethos", name="binary"),
    preprocess_steps=[
        Shuffle(page_size=1000000),
        SplitRandomMix({"train": "train[20%]", "test": "train[80%]"}),
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
    __tags__={
        "Hate Speech Detection": True,
        "annotations_creators": ["crowdsourced", "expert-generated"],
        "arxiv": "2006.08328",
        "croissant": True,
        "language": "en",
        "language_creators": ["found", "other"],
        "license": "agpl-3.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "n<1K",
        "source_datasets": "original",
        "task_categories": "text-classification",
        "task_ids": ["multi-label-classification", "sentiment-classification"],
    },
)

test_card(card, demos_pool_size=20, loader_limit=1000)
add_to_catalog(card, "cards.ethos_binary", overwrite=True)
