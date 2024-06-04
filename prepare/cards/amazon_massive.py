from datasets import get_dataset_config_names, load_dataset_builder
from unitxt.blocks import (
    AddFields,
    LoadHF,
    MapInstanceValues,
    RenameFields,
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
            RenameFields(field_to_field={"utt": "text", "intent": "label"}),
            AddFields(
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
            "flags": ["croissant", "natural-language-understanding"],
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
            "MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations for the Natural Language Understanding tasks of intent prediction and slot annotation. Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing the SLURP dataset, composed of general Intelligent Voice Assistant single-shot interactions. See the full description on the dataset page: https://huggingface.co/datasets/AmazonScience/massive.\n"
        ),
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    filename = lang.replace("-", "_")
    add_to_catalog(card, f"cards.amazon_mass.{filename}", overwrite=True)
