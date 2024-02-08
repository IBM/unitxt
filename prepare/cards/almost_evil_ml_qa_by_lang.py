import json

from src.unitxt.blocks import FormTask, LoadHF, RenameFields, SplitRandomMix, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import (
    Apply,
    CopyFields,
    FilterByCondition,
    ListFieldValues,
    RemoveFields,
)
from src.unitxt.test_utils.card import test_card

langs = ["en", "de", "it", "fr", "es", "ru", "nl", "pt"]
# Counter({'en': 1995, 'de': 2302, 'it': 2210, 'fr': 2156, 'es': 2090, 'ru': 2058, 'nl': 2017, 'pt': 1994})

for lang in langs:
    card = TaskCard(
        loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
        preprocess_steps=[
            Apply("METADATA", function=json.loads, to_field="metadata"),
            CopyFields(
                field_to_field=[("metadata/language", "extracted_language")],
                use_query=True,
            ),
            FilterByCondition(values={"extracted_language": lang}, condition="eq"),
            RemoveFields(fields=["extracted_language", "metadata"]),
            SplitRandomMix(
                {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
            ),
            RenameFields(field_to_field={"INSTRUCTION": "question"}),
            ListFieldValues(fields=["RESPONSE"], to_field="answers"),
        ],
        task=FormTask(
            inputs=["question"],
            outputs=["answers"],
            metrics=["metrics.rouge"],
        ),
        templates="templates.qa.open.all",
    )

    if lang == langs[0]:
        test_card(card, debug=False)
    add_to_catalog(card, f"cards.almostEvilML_qa_by_lang.{lang}", overwrite=True)
