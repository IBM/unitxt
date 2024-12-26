import json

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadHF
from unitxt.operators import CastFields, Copy, ExecuteExpression, ListFieldValues, Set
from unitxt.splitters import SplitRandomMix
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="enelpol/rag-mini-bioasq",
        name="question-answer-passages",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
            },
        ),
        ExecuteExpression(
            expression="[str(d) for d in relevant_passage_ids]",
            to_field="reference_context_ids",
        ),
        Set(
            fields={
                "reference_contexts": [],
                "is_answerable_label": True,
                "metadata_field": "",
            }
        ),
        ListFieldValues(
            fields=["answer"],
            to_field="reference_answers",
        ),
    ],
    task="tasks.rag.end_to_end",
    # templates=["templates.empty"],
    templates={"default": "templates.rag.end_to_end.json_predictions"},
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

add_to_catalog(
    card,
    "cards.rag.benchmark.bioasq.en",
)


# Documents
card = TaskCard(
    loader=LoadHF(
        path="enelpol/rag-mini-bioasq",
        name="text-corpus",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            {
                "train": "test[100%]",
            }
        ),
        CastFields(fields={"id": "str"}),
        Copy(
            field_to_field={
                "id": "document_id",
            },
        ),
        ListFieldValues(
            fields=["passage"],
            to_field="passages",
        ),
        Set(
            fields={
                "metadata_field": "",
                "title": "",
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

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.bioasq.en", overwrite=True)
