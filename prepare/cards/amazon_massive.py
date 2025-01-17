from datasets import get_dataset_config_names, load_dataset_builder
from unitxt.blocks import (
    LoadHF,
    MapInstanceValues,
    Rename,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

langs = get_dataset_config_names("AmazonScience/massive", trust_remote_code=True)
# now langs is the list of all languages showing in the dataset


ds_builder = load_dataset_builder("AmazonScience/massive", trust_remote_code=True)
classlabels = ds_builder.info.features["intent"]

mappers = {}
for i in range(len(classlabels.names)):
    mappers[str(i)] = classlabels.names[i]


for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="AmazonScience/massive", name=lang),
        preprocess_steps=[
            MapInstanceValues(mappers={"intent": mappers}),
            Rename(field_to_field={"utt": "text", "intent": "label"}),
            Set(
                fields={
                    "classes": classlabels.names,
                    "text_type": "sentence",
                    "type_of_class": "intent",
                }
            ),
        ],
        task="tasks.classification.multi_class",
        templates="templates.classification.multi_class.all",
        __tags__={
            "annotations_creators": "expert-generated",
            "arxiv": "2204.08582",
            "flags": ["natural-language-understanding"],
            "language_creators": "found",
            "license": "cc-by-4.0",
            "multilinguality": [
                "af-ZA",
                "am-ET",
                "ar-SA",
                "az-AZ",
                "bn-BD",
                "ca-ES",
                "cy-GB",
                "da-DK",
                "de-DE",
                "el-GR",
                "en-US",
                "es-ES",
                "fa-IR",
                "fi-FI",
                "fr-FR",
                "he-IL",
                "hi-IN",
                "hu-HU",
                "hy-AM",
                "id-ID",
                "is-IS",
                "it-IT",
                "ja-JP",
                "jv-ID",
                "ka-GE",
                "km-KH",
                "kn-IN",
                "ko-KR",
                "lv-LV",
                "ml-IN",
                "mn-MN",
                "ms-MY",
                "my-MM",
                "nb-NO",
                "nl-NL",
                "pl-PL",
                "pt-PT",
                "ro-RO",
                "ru-RU",
                "sl-SL",
                "sq-AL",
                "sv-SE",
                "sw-KE",
                "ta-IN",
                "te-IN",
                "th-TH",
                "tl-PH",
                "tr-TR",
                "ur-PK",
                "vi-VN",
                "zh-CN",
                "zh-TW",
            ],
            "region": "us",
            "size_categories": "100K<n<1M",
            "source_datasets": "original",
            "task_categories": "text-classification",
            "task_ids": ["intent-classification", "multi-class-classification"],
        },
        __description__=(
            "Amazon's MASSIVE is a dataset with over 1M utterances across 52 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. It covers 60 intents and 55 slot types, created by localizing dataset for voice assistant interactions."
        ),
        __short_description__=(
            "Uses a dataset of over 1M utterances from interactions with Amazon's voice assistant that is localized into 52 languages and annotated with intent and slot type information to evaluate a model's ability to classify multilingual text. Metric shows the F1 score."
        ),
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    filename = lang.replace("-", "_")
    add_to_catalog(card, f"cards.amazon_mass.{filename}", overwrite=True)
