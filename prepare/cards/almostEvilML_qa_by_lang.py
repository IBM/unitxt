import json

from src.unitxt.blocks import FormTask, LoadHF, RenameFields, SplitRandomMix, TaskCard
from src.unitxt.logging_utils import get_logger
from src.unitxt.operators import (
    Apply,
    CopyFields,
    FilterByCondition,
    GlobalRougeOnly,
    ListFieldValues,
    Perturbate,
    RemoveFields,
    StreamRefiner,
)
from src.unitxt.standard import NaiveRecipe
from src.unitxt.text_utils import print_dict

langs = ["en", "de", "it", "fr", "es", "ru", "nl", "pt"]
# Counter({'en': 1995, 'de': 2302, 'it': 2210, 'fr': 2156, 'es': 2090, 'ru': 2058, 'nl': 2017, 'pt': 1994})

logger = get_logger()

for lang in ["en"]:
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
            StreamRefiner(max_instances=10),
            SplitRandomMix(
                {"train": "train[90%]", "validation": "train[5%]", "test": "train[5%]"}
            ),
            RenameFields(field_to_field={"INSTRUCTION": "question"}),
            ListFieldValues(fields=["RESPONSE"], to_field="answers"),
            FormTask(
                inputs=["question"],
                outputs=["answers"],
                metrics=["metrics.rouge"],
            ),
            "templates.qa.open.simple2",
            Perturbate(
                field="target",
                to_field="prediction",
                percentage_to_perturbate=30,
            ),
            # increase the noise
            Perturbate(
                field="prediction",
                percentage_to_perturbate=30,
            ),
            GlobalRougeOnly(
                pred_field_name="prediction",
                refs_field_name="references",
            ),
        ],
    )
    naive_recipe = NaiveRecipe(card=card)

    multi_stream = naive_recipe()

    for stream_name, stream in multi_stream.items():
        logger.info(f"Stream: {stream_name}")
        for instance in stream:
            print_dict(instance)

    # if lang == "en":
    #     test_card(card, debug=False)
    # add_to_catalog(card, f"cards.almostEvilML_qa_by_lang.{lang}", overwrite=True)
