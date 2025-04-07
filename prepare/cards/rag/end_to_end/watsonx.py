import json

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.loaders import LoadHF
from unitxt.operators import Copy
from unitxt.splitters import RenameSplits
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="ibm-research/watsonxDocsQA",
        name="question_answers",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                "question": "question",
                "question_id": "question_id",
            },
        ),
        Wrap(
            field="correct_answer_document_ids",
            inside="list",
            to_field="reference_context_ids",
        ),
        Wrap(
            field="correct_answer",
            inside="list",
            to_field="reference_answers",
        ),
    ],
    task="tasks.rag.end_to_end",
    templates={"default": "templates.rag.end_to_end.json_predictions"},
    __tags__={"license": "Apache 2.0", "url":"https://huggingface.co/datasets/ibm-research/watsonxDocsQA"},
    __description__="""TO DEFINE
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

add_to_catalog(card, "cards.rag.benchmark.watsonxqa.en", overwrite=True)


# Documents
card = TaskCard(
    loader=LoadHF(
        path="ibm-research/watsonxDocsQA",
        name="corpus",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        RenameSplits({"test": "train"}),
        Copy(field="doc_id", to_field="document_id"),
        Wrap(field="document", inside="list", to_field="passages"),
    ],
    task="tasks.rag.corpora",
    templates={
        "empty": InputOutputTemplate(
            input_format="",
            output_format="",
        ),
    },
    __tags__={"license": "Apache 2.0", "url" : "https://huggingface.co/datasets/ibm-research/watsonxDocsQA"},
    __description__="""TO DEFINE
""",
)

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.watsonxqa.en", overwrite=True)
