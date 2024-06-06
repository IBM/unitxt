from unitxt.blocks import LoadHF, RenameFields, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    Copy,
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
            Copy(field="metadata/language", to_field="extracted_language"),
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
            "flags": ["QnA", "wikihow"],
            "language": ["en", "ru", "pt", "it", "es", "fr", "de", "nl"],
            "license": "cc-by-nc-3.0",
            "region": "us",
            "size_categories": "10K<n<100K",
            "task_categories": "question-answering",
        },
        __description__=(
            "Contains Parquet of a list of instructions and WikiHow articles on different languages. See the full description (and warnings) on the dataset page: https://huggingface.co/datasets/0x22almostEvil/multilingual-wikihow-qa-16k."
        ),
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.almost_evil.{lang}", overwrite=True)
