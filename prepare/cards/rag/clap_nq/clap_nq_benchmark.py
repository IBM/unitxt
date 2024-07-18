from dataclasses import dataclass

from unitxt import add_to_catalog
from unitxt.blocks import TaskCard, TemplatesDict
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, ListFieldValues, Set
from unitxt.test_utils.card import test_card

from prepare.tasks.rag.rag_task import (
    TaskRagEndToEndConstants,
    TaskRagEndToEndInputConstants,
    TaskRagEndToEndReferenceConstants,
)


@dataclass(frozen=True)
class ClapNqBenchmarkConstants:
    # Raw_data
    TRAIN_RAW_FILE_URL: str = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/train/question_train_answerable.tsv"
    TEST_RAW_FILE_URL: str = "https://raw.githubusercontent.com/primeqa/clapnq/main/retrieval/dev/question_dev_answerable.tsv"

    # Fields
    ID: str = "id"
    QUESTION: str = "question"
    DOC_ID_LIST: str = "doc-id-list"
    ANSWERS: str = "answers"

    ARTIFACT_NAME: str = "cards.rag.benchmark.clap_nq.en"


card = TaskCard(
    loader=LoadCSV(
        sep="\t",
        files={
            "train": ClapNqBenchmarkConstants.TRAIN_RAW_FILE_URL,
            "test": ClapNqBenchmarkConstants.TEST_RAW_FILE_URL,
        },
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                ClapNqBenchmarkConstants.QUESTION: TaskRagEndToEndInputConstants.QUESTION,
                ClapNqBenchmarkConstants.ID: TaskRagEndToEndInputConstants.QUESTION_ID,
            },
        ),
        Set(
            fields={
                TaskRagEndToEndReferenceConstants.REFERENCE_CONTEXTS: [],
                TaskRagEndToEndReferenceConstants.IS_ANSWERABLE_LABEL: True,
                TaskRagEndToEndInputConstants.METADATA_FIELD: "",
            }
        ),
        ListFieldValues(
            fields=[ClapNqBenchmarkConstants.DOC_ID_LIST],
            to_field=TaskRagEndToEndReferenceConstants.REFERENCE_CONTEXT_IDS,
        ),
        ListFieldValues(
            fields=[ClapNqBenchmarkConstants.ANSWERS],
            to_field=TaskRagEndToEndReferenceConstants.REFERENCE_ANSWERS,
        ),
    ],
    task=TaskRagEndToEndConstants.TASKS_RAG_END_TO_END,
    templates=TemplatesDict(
        {"default": TaskRagEndToEndConstants.TEMPLATE_RAG_END_TO_END_JSON_PREDICTIONS}
    ),
)

test_card(
    card,
    strict=True,
    debug=False,
    demos_taken_from="test",
    demos_pool_size=5,
)
add_to_catalog(card, ClapNqBenchmarkConstants.ARTIFACT_NAME, overwrite=True)
