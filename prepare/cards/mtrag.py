import json

from unitxt import add_to_catalog
from unitxt.blocks import (
    TaskCard,
)
from unitxt.collections_operators import Dictify, Wrap
from unitxt.loaders import LoadCSV
from unitxt.operators import (
    Cast,
    Copy,
    MapInstanceValues,
    Set,
    ZipFieldValues,
)
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

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
                "task_id": "question_id",
                "contexts/*/document_id": "reference_context_ids",
                "contexts/*/text": "reference_contexts",
                "input/*/speaker": "roles",
                "input/*/text": "contents",
            },
        ),
        ZipFieldValues(
            fields=["roles", "contents"],
            to_field="conversation",
        ),
        Dictify(
            field="conversation",
            with_keys=["role", "content"],
            to_field="question",
            process_every_value=True,
        ),
    ],
    task="tasks.rag.end_to_end",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "apache-2.0"},
    __description__="""MTRAG: a comprehensive and diverse human-generated multi-turn RAG dataset, accompanied by four document corpora. To the best of our knowledge, MTRAG is the first end-to-end human-generated multi-turn RAG benchmark that reflects real-world properties of multi-turn conversations.
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
    strict=False,
    full_mismatch_prediction_values=[json.dumps(wrong_answer)],
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)

add_to_catalog(card, "cards.rag.mtrag", overwrite=True)


for subset in ["clapnq", "cloud", "fiqa", "govt"]:
    subset_operators = []
    if subset in ["fiqa", "clapnq"]:
        subset_operators.append(
            Cast(
                field="_id",
                to="str",
                to_field="document_id",
            )
        )
    if subset in ["cloud"]:
        subset_operators.append(Set(fields={"title": ""}))

    card = TaskCard(
        loader=LoadCSV(
            files={
                "test": f"https://github.com/IBM/mt-rag-benchmark/raw/refs/heads/main/corpora/{subset}.jsonl.zip"
            },
            compression="zip",
            file_type="json",
            lines=True,
            data_classification_policy=["public"],
        ),
        preprocess_steps=[
            *subset_operators,
            Wrap(field="text", inside="list", to_field="passages"),
            Set(
                fields={
                    "metadata_field": "",
                }
            ),
        ],
        task="tasks.rag.corpora",
        templates={
            "empty": InputOutputTemplate(
                input_format="",
                output_format="",
            ),
        },
    )
    test_card(
        card,
        strict=False,
        demos_taken_from="test",
    )

    add_to_catalog(card, f"cards.rag.mtrag.documents.{subset}", overwrite=True)
