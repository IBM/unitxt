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
    __tags__={"license": "Apache License 2.0", "url": "https://huggingface.co/datasets/PrimeQA/clapnq"},
    __description__="""CLAP NQ is created from the subset of Natural Questions (NQ) that have a long answer but no short answer. NQ consists of ~380k examples. There are ~30k questions that are long answers without short answers excluding tables and lists. To increases the likelihood of longer answers we only explored ones that have more than 5 sentences in the passage. The subset that was annotated consists of ~12k examples. All examples where cohesion of non-consecutive sentences was required for the answer were annotated a second time. The final dataset is made up of all data that went through two rounds of annotation. (We provide the single round annotations as well - it is only training data) An equal amount of unanswerable questions have also been added from the original NQ train/dev sets. Details about the annotation task and unanswerables can be found at https://github.com/primeqa/clapnq/blob/main/annotated_data.""",
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
                "metadata_field": {},
            }
        ),
    ],
    task="tasks.rag.corpora",
    __tags__={"license": "Apache License 2.0", "url":"https://huggingface.co/datasets/PrimeQA/clapnq"},
    __description__="""CLAP NQ is created from the subset of Natural Questions (NQ) that have a long answer but no short answer. NQ consists of ~380k examples. There are ~30k questions that are long answers without short answers excluding tables and lists. To increases the likelihood of longer answers we only explored ones that have more than 5 sentences in the passage. The subset that was annotated consists of ~12k examples. All examples where cohesion of non-consecutive sentences was required for the answer were annotated a second time. The final dataset is made up of all data that went through two rounds of annotation. (We provide the single round annotations as well - it is only training data) An equal amount of unanswerable questions have also been added from the original NQ train/dev sets. Details about the annotation task and unanswerables can be found at https://github.com/primeqa/clapnq/blob/main/annotated_data.""",
    templates={
        "empty": InputOutputTemplate(
            input_format="",
            output_format="",
        ),
    },
)

# Not testing card, because documents are not evaluated.
add_to_catalog(card, "cards.rag.documents.clap_nq.en", overwrite=True)
