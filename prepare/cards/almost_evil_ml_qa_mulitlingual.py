import json

from unitxt.blocks import LoadHF, RenameFields, SplitRandomMix, TaskCard
from unitxt.catalog import add_to_catalog
from unitxt.operators import (
    Apply,
    CopyFields,
    FilterByCondition,
    ListFieldValues,
    RemoveFields,
)
from unitxt.test_utils.card import test_card

langs = ["en", "de", "it", "fr", "es", "ru", "nl", "pt"]
# Counter({'en': 1995, 'de': 2302, 'it': 2210, 'fr': 2156, 'es': 2090, 'ru': 2058, 'nl': 2017, 'pt': 1994})

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
        preprocess_steps=[
            Apply("METADATA", function=json.loads, to_field="metadata"),
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
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.almost_evil.{lang}", overwrite=True)
