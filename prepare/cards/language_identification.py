from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import MapInstanceValues, Rename
from unitxt.test_utils.card import test_card

language_codes = {
    "ar": "arabic",
    "bg": "bulgarian",
    "de": "german",
    "el": "modern greek",
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "hi": "hindi",
    "it": "italian",
    "ja": "japanese",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sw": "swahili",
    "th": "thai",
    "tr": "turkish",
    "ur": "urdu",
    "vi": "vietnamese",
    "zh": "chinese",
}

card = TaskCard(
    loader=LoadHF(path="papluca/language-identification"),
    preprocess_steps=[
        Rename(field_to_field={"labels": "label"}),
        MapInstanceValues(mappers={"label": language_codes}),
    ],
    task="tasks.language_identification",
    templates="templates.language_identification.all",
    __description__=(
        "The Language Identification dataset is a collection of 90k samples consisting of text passages and corresponding language label. This dataset was created by collecting data from 3 sources: Multilingual Amazon Reviews Corpus, XNLI, and STSb Multi MT. See the full description on the dataset page: https://huggingface.co/datasets/papluca/language-identification."
    ),
)

test_card(card, strict=False)
add_to_catalog(card, "cards.language_identification", overwrite=True)
