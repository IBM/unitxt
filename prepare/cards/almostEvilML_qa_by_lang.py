from src.unitxt.blocks import FormTask, LoadHF, RenameFields, SplitRandomMix, TaskCard
from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import Apply, FilterByValues, RemoveFields
from src.unitxt.test_utils.card import test_card

langs = ["de", "it", "fr", "es", "ru", "nl", "en", "pt"]
# Counter({'de': 2302, 'it': 2210, 'fr': 2156, 'es': 2090, 'ru': 2058, 'nl': 2017, 'en': 1995, 'pt': 1994})


def extract_lang_from_METADATA(md_as_string: str) -> str:
    import json

    md_as_json = json.loads(md_as_string)
    return md_as_json["language"]


for lang in langs:

    card = TaskCard(
        loader=LoadHF(path="0x22almostEvil/multilingual-wikihow-qa-16k"),
        preprocess_steps=[
            Apply("METADATA", function=extract_lang_from_METADATA, to_field="extracted_language"),
            FilterByValues(required_values={"extracted_language": lang}),
            RemoveFields(fields=["extracted_language"]),
            SplitRandomMix({"train": "train[80%]", "validation": "train[10%]", "test": "train[10%]"}),
            RenameFields(field_to_field={"INSTRUCTION": "question", "RESPONSE": "answer"}),
        ],
        task=FormTask(
            inputs=["question"],
            outputs=["answer"],
            metrics=["metrics.rouge"],
        ),
        templates="templates.qa.open.all",
    )

    if lang == langs[0]:
        test_card(card, debug=False, loader_limit=16000)
        # because most of the records loaded by the loader are screened out by language,
        # and langs are not randomly spread over the dataset
    add_to_catalog(card, f"cards.almostEvilML_qa.{lang}", overwrite=True)
