from unitxt.blocks import LoadHF, RenameFields, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    CopyFields,
    FilterByCondition,
    ListFieldValues,
    RemoveFields,
)
from unitxt.struct_data_operators import LoadJson
from unitxt.test_utils.card import test_card

langs = ["en", "de", "it", "fr", "es", "ru", "nl", "pt"]
# Counter({'en': 1995, 'de': 2302, 'it': 2210, 'fr': 2156, 'es': 2090, 'ru': 2058, 'nl': 2017, 'pt': 1994})

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
        preprocess_steps=[
            LoadJson(field="METADATA", to_field="metadata"),
            CopyFields(field_to_field=[("metadata/language", "extracted_language")]),
            FilterByCondition(values={"extracted_language": lang}, condition="eq"),
            RemoveFields(fields=["extracted_language", "metadata"]),
            SplitRandomMix(
                {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
            ),
            RenameFields(field_to_field={"INSTRUCTION": "question"}),
            ListFieldValues(fields=["RESPONSE"], to_field="answers"),
        ],
        task="tasks.qa.open",
        templates="templates.qa.open.all",
        __tags__={
            "dataset_info_tags": [
                "task_categories:question-answering",
                "size_categories:10K<n<100K",
                "language:en",
                "language:ru",
                "language:pt",
                "language:it",
                "language:es",
                "language:fr",
                "language:de",
                "language:nl",
                "license:cc-by-nc-3.0",
                "wikihow",
                "QnA",
                "croissant",
                "region:us",
            ]
        },
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.almost_evil.{lang}", overwrite=True)
