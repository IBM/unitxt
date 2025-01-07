import json
from dataclasses import dataclass

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard
from unitxt.loaders import LoadCSV, LoadHF
from unitxt.operators import Copy, ListFieldValues, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card


@dataclass(frozen=True)
class ClapNqBenchmark:
    # Raw_data
    TRAIN_RAW_FILE_URL: str = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/train/question_train_answerable.tsv"
    TEST_RAW_FILE_URL: str = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv"


card = TaskCard(
    loader=LoadCSV(
        sep="\t",
        files={
            "train": ClapNqBenchmark.TRAIN_RAW_FILE_URL,
            "test": ClapNqBenchmark.TEST_RAW_FILE_URL,
        },
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                "question": "question",
                "id": "question_id",
            },
        ),
        Set(
            fields={
                "reference_contexts": [],
                "is_answerable_label": True,
                "metadata_field": "",
            }
        ),
        ListFieldValues(
            fields=["doc-id-list"],
            to_field="reference_context_ids",
        ),
        ListFieldValues(
            fields=["answers"],
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

add_to_catalog(card, "cards.rag.benchmark.clap_nq.en", overwrite=True)

# Documents
card = TaskCard(
    loader=LoadHF(
        path="PrimeQA/clapnq_passages",
        name="train",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                "id": "document_id",
                "title": "title",
            },
        ),
        ListFieldValues(
            fields=["text"],
            to_field="passages",
        ),
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

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.clap_nq.en", overwrite=True)
