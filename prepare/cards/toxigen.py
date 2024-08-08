from unitxt import add_to_catalog
from unitxt.blocks import LoadHF, TaskCard
from unitxt.operators import MapInstanceValues, Rename, Set, Shuffle
from unitxt.splitters import SplitRandomMix
from unitxt.templates import InputOutputTemplate, TemplatesList
from unitxt.test_utils.card import test_card

""" In order to use this card, you must have a HF login token.
First, sign up for an account on the HF portal and accept the terms and conditions related to the toxigen dataset.
After that, incorporate your HF token into the environment settings by adding HUGGING_FACE_HUB_TOKEN=<your HF token>.
"""

card = TaskCard(
    loader=LoadHF(path="skg/toxigen-data", name="train"),
    preprocess_steps=[
        Shuffle(page_size=251000),
        SplitRandomMix({"train": "train[20%]", "test": "train[80%]"}),
        MapInstanceValues(mappers={"prompt_label": {"0": "not toxic", "1": "toxic"}}),
        Rename(field_to_field={"prompt": "text"}),
        Rename(field_to_field={"prompt_label": "label"}),
        Set(
            fields={
                "classes": ["not toxic", "toxic"],
                "type_of_class": "toxicity",
            }
        ),
    ],
    task="tasks.classification.multi_class",
    templates=TemplatesList(
        [
            InputOutputTemplate(
                input_format="Given this {text_type}: {text}. Classify if it contains {type_of_class}. classes: {"
                "classes}. I would classify this {text_type} as: ",
                output_format="{label}",
                postprocessors=[
                    "processors.take_first_non_empty_line",
                    "processors.toxic_or_not_toxic",
                ],
            ),
        ]
    ),
    __description__=(
        "This dataset is for implicit hate speech detection. All instances were generated using GPT-3 and the methods described in our paper. Languages All text is written in English. Dataset Structure Data Fields We release TOXIGEN as a dataframe with the following fields: prompt is the prompt used for… See the full description on the dataset page: https://huggingface.co/datasets/toxigen/toxigen-data."
    ),
)

try:
    test_card(card)
except Exception:
    pass

add_to_catalog(card, "cards.toxigen", overwrite=True)
