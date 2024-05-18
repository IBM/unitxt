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

langs = get_dataset_config_names("AmazonScience/massive")
# now langs is the list of all languages showing in the dataset


ds_builder = load_dataset_builder("AmazonScience/massive")
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
        __description__=(
            "MASSIVE is a parallel dataset of > 1M utterances across 51 languages with annotations"
            "for the Natural Language Understanding tasks of intent prediction and slot annotation."
            "Utterances span 60 intents and include 55 slot types. MASSIVE was created by localizing"
            "the SLURP dataset, composed of general Intelligent Voice Assistant single-shot interactions."
        ),
        __tags__={
            "dataset_info_tags": [
                "task_categories:text-classification",
                "task_ids:intent-classification",
                "task_ids:multi-class-classification",
                "annotations_creators:expert-generated",
                "language_creators:found",
                "multilinguality:af-ZA",
                "multilinguality:am-ET",
                "multilinguality:ar-SA",
                "multilinguality:az-AZ",
                "multilinguality:bn-BD",
                "multilinguality:ca-ES",
                "multilinguality:cy-GB",
                "multilinguality:da-DK",
                "multilinguality:de-DE",
                "multilinguality:el-GR",
                "multilinguality:en-US",
                "multilinguality:es-ES",
                "multilinguality:fa-IR",
                "multilinguality:fi-FI",
                "multilinguality:fr-FR",
                "multilinguality:he-IL",
                "multilinguality:hi-IN",
                "multilinguality:hu-HU",
                "multilinguality:hy-AM",
                "multilinguality:id-ID",
                "multilinguality:is-IS",
                "multilinguality:it-IT",
                "multilinguality:ja-JP",
                "multilinguality:jv-ID",
                "multilinguality:ka-GE",
                "multilinguality:km-KH",
                "multilinguality:kn-IN",
                "multilinguality:ko-KR",
                "multilinguality:lv-LV",
                "multilinguality:ml-IN",
                "multilinguality:mn-MN",
                "multilinguality:ms-MY",
                "multilinguality:my-MM",
                "multilinguality:nb-NO",
                "multilinguality:nl-NL",
                "multilinguality:pl-PL",
                "multilinguality:pt-PT",
                "multilinguality:ro-RO",
                "multilinguality:ru-RU",
                "multilinguality:sl-SL",
                "multilinguality:sq-AL",
                "multilinguality:sv-SE",
                "multilinguality:sw-KE",
                "multilinguality:ta-IN",
                "multilinguality:te-IN",
                "multilinguality:th-TH",
                "multilinguality:tl-PH",
                "multilinguality:tr-TR",
                "multilinguality:ur-PK",
                "multilinguality:vi-VN",
                "multilinguality:zh-CN",
                "multilinguality:zh-TW",
                "size_categories:100K<n<1M",
                "source_datasets:original",
                "license:cc-by-4.0",
                "natural-language-understanding",
                "croissant",
                "arxiv:2204.08582",
                "region:us",
            ]
        },
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    filename = lang.replace("-", "_")
    add_to_catalog(card, f"cards.amazon_mass.{filename}", overwrite=True)
