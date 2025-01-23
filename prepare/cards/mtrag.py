import json
from typing import Any, Dict

from unitxt import add_to_catalog
from unitxt.blocks import (
    TaskCard,
)
from unitxt.collections_operators import Dictify
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, InstanceOperator, MapInstanceValues, ZipFieldValues
from unitxt.test_utils.card import test_card


class TempOperator(InstanceOperator):
    def process(
        self, instance: Dict[str, Any], stream_name: str | None = None
    ) -> Dict[str, Any]:
        return instance


card = TaskCard(
    loader=LoadCSV(
        files={
            "test": "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/refs/heads/main/human/generation_tasks/reference+RAG.jsonl"
        },
        file_type="json",
        lines=True,
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        MapInstanceValues(
            {
                "Answerability": {
                    "['UNANSWERABLE']": False,
                    "['ANSWERABLE']": True,
                    "['PARTIAL']": True,
                },
            }
        ),
        Copy(
            field_to_field={
                "targets/*/text": "reference_answers",
                "Answerability": "is_answerable_label",
                "task_id": "turn_id",
                "contexts/*/document_id": "reference_context_ids",
                "contexts/*/text": "reference_contexts",
                "input/*/speaker": "roles",
                "input/*/text": "contents",
                "input/0/text": "question",
            },
        ),
        ZipFieldValues(
            fields=["roles", "contents"],
            to_field="conversation",
        ),
        Dictify(
            field="conversation",
            with_keys=["role", "content"],
            process_every_value=True,
        ),
    ],
    task="tasks.rag.end_to_end.multi_turn",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "cc-by-2.5"},
    __description__="""This dataset is a subset of a training dataset by the BioASQ Challenge, which is available here.

It is derived from rag-datasets/rag-mini-bioasq.

Modifications include:

filling in missing passages (some of them contained "nan" instead of actual text),
changing relevant_passage_ids' type from string to sequence of ints,
deduplicating the passages (removed 40 duplicates) and fixing the relevant_passage_ids in QAP triplets to point to the corrected, deduplicated passages' ids,
splitting QAP triplets into train and test splits.
""",
)
wrong_answer = {
    "contexts": ["hi"],
    "is_answerable": True,
    "answer": "Don't know",
    "context_ids": ["id0"],
}

test_card(
    card,
    strict=True,
    full_mismatch_prediction_values=[json.dumps(wrong_answer)],
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)

add_to_catalog(card, "cards.rag.benchmark.bioasq.en", overwrite=True)
exit()

# for subset in [
#     "clapnq",
#     # "cloud",
#     "fiqa",
#     "govt"
#     ]:

#     subset_operators = []
#     if subset in ["fiqa", "clapnq"]:
#         subset_operators.append(
#             Cast(
#                 field="_id",
#                 to="str",
#                 to_field="document_id",
#             )
#         )

#     card = TaskCard(
#         loader=LoadCSV(
#             files={'test': f'https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/{subset}.jsonl.zip'},
#             compression="zip",
#             file_type="json",
#             lines=True,
# data_classification_policy=["public"]

#         ),
#         preprocess_steps=[
#             *subset_operators,
#             Wrap(field="text", inside="list", to_field="passages"),
#             Set(
#                 fields={
#                     "metadata_field": "",
#                 }
#             ),
#         ],
#         task="tasks.rag.corpora",
#         templates={
#             "empty": InputOutputTemplate(
#                 input_format="",
#                 output_format="",
#             ),
#         },
#     )
#     test_card(
#         card,
#         strict=False,
#         demos_taken_from="test",
#     )

#     add_to_catalog(
#         card, f"cards.rag.mtrag.documents.{subset}", overwrite=True
#     )
