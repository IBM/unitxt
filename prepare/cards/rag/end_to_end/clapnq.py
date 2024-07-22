import itertools
import math
import random
from dataclasses import dataclass

import pandas as pd
from unitxt import add_to_catalog
from unitxt.blocks import TaskCard, TemplatesDict
from unitxt.loaders import LoadCSV, LoadFromDictionary
from unitxt.operators import Copy, ListFieldValues, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card


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


@dataclass(frozen=True)
class ClapNqDocumentsConstants:
    # Raw_data
    RAW_FILE_URL: str = "https://media.githubusercontent.com/media/primeqa/clapnq/main/retrieval/passages.tsv"

    # Fields
    ID: str = "id"
    TEXT: str = "text"
    TITLE: str = "title"

    ARTIFACT_NAME: str = "cards.rag.documents.clap_nq.en"


@dataclass(frozen=True)
class SubsetCreationConstants:
    # For subset creation
    SUBSET_SIZE = 100
    RANDOM_SEED = 43
    NEGATIVE_DOCUMENTS_FACTOR = 4


# We want to create a subset of ClapNq - based on a subset of questions from the benchmark.
random.seed(SubsetCreationConstants.RANDOM_SEED)
# 1 - first we read the benchmark and the documents
benchmark_train_df = pd.read_csv(ClapNqBenchmarkConstants.TRAIN_RAW_FILE_URL, sep="\t")
benchmark_test_df = pd.read_csv(ClapNqBenchmarkConstants.TEST_RAW_FILE_URL, sep="\t")
documents_df = pd.read_csv(ClapNqDocumentsConstants.RAW_FILE_URL, sep="\t")

# 2 - We compute the size of train subset and test subset
ratio_train_test = len(benchmark_train_df) / len(benchmark_test_df)
train_size = math.floor(SubsetCreationConstants.SUBSET_SIZE / ratio_train_test)
test_size = SubsetCreationConstants.SUBSET_SIZE - train_size

# 3 - We create the reduced dataframes
reduced_train_benchmark_df = benchmark_train_df.sample(
    n=train_size, random_state=SubsetCreationConstants.RANDOM_SEED
)
reduced_test_benchmark_df = benchmark_test_df.sample(
    n=test_size, random_state=SubsetCreationConstants.RANDOM_SEED
)

# 4 - We get all the gt_doc_ids:
benchmark_doc_id = set(
    reduced_train_benchmark_df[ClapNqBenchmarkConstants.DOC_ID_LIST].tolist()
) | set(reduced_test_benchmark_df[ClapNqBenchmarkConstants.DOC_ID_LIST].tolist())
benchmark_doc_id = list(set(itertools.chain(*benchmark_doc_id)))

# 5 - We get all the doc_ids:
total_doc_ids = documents_df["id"].tolist()
# 5a - We remove duplicates and we remove the benchmark_doc_ids
other_doc_ids = list(set(total_doc_ids) - set(benchmark_doc_id))
# 5b - We shuffle them according to the seed
# 5b1 - First we sort
other_doc_ids.sort()
# 5b2 : we shuffle
random.shuffle(other_doc_ids)

# 6 - We take the number of doc_ids
other_docs_ids_size = SubsetCreationConstants.NEGATIVE_DOCUMENTS_FACTOR * len(
    benchmark_doc_id
)
other_doc_ids = other_doc_ids[:other_docs_ids_size]

# 7 - we merge the two sets:
all_sampled_docs_ids = set(benchmark_doc_id) | set(other_doc_ids)

# 8 - we select the documents:
reduced_documents_df = documents_df[documents_df["id"].isin(all_sampled_docs_ids)]
reduced_documents_df = reduced_documents_df.reset_index(drop=True)

# 9 - We transform the dataframes to list of dicts
reduced_train_list_of_dicts = reduced_train_benchmark_df.to_dict(orient="records")
reduced_test_list_of_dicts = reduced_test_benchmark_df.to_dict(orient="records")
reduced_documents_list_of_dicts = reduced_documents_df.to_dict(orient="records")


full_benchmark_loader = LoadCSV(
    sep="\t",
    files={
        "train": ClapNqBenchmarkConstants.TRAIN_RAW_FILE_URL,
        "test": ClapNqBenchmarkConstants.TEST_RAW_FILE_URL,
    },
)

reduced_benchmark_loader = (
    LoadFromDictionary(
        data={"train": reduced_train_list_of_dicts, "test": reduced_test_list_of_dicts},
    ),
)

for loader, card_name in zip(
    [full_benchmark_loader, reduced_benchmark_loader],
    ["cards.rag.benchmark.clap_nq.en", "cards.rag.benchmark.clap_nq_reduced.en"],
):
    card = TaskCard(
        loader=loader,
        preprocess_steps=[
            Copy(
                field_to_field={
                    ClapNqBenchmarkConstants.QUESTION: "question",
                    ClapNqBenchmarkConstants.ID: "question_id",
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
                fields=[ClapNqBenchmarkConstants.DOC_ID_LIST],
                to_field="reference_context_ids",
            ),
            ListFieldValues(
                fields=[ClapNqBenchmarkConstants.ANSWERS],
                to_field="reference_answers",
            ),
        ],
        task="tasks.rag.end_to_end",
        templates=TemplatesDict(
            {"default": "templates.rag.end_to_end.json_predictions"}
        ),
    )

    test_card(
        card,
        strict=True,
        debug=False,
        demos_taken_from="test",
        demos_pool_size=5,
    )

    add_to_catalog(card, card_name, overwrite=True)


full_documents_loader = LoadCSV(
    sep="\t",
    files={"train": ClapNqDocumentsConstants.RAW_FILE_URL},
)
reduced_documents_loader = LoadFromDictionary(
    data={"train": reduced_documents_list_of_dicts}
)


# Documents
for loader, card_name in zip(
    [full_documents_loader, reduced_documents_loader],
    ["cards.rag.documents.clap_nq.en", "cards.rag.documents.clap_nq_reduced.en"],
):
    card = TaskCard(
        loader=loader,
        preprocess_steps=[
            Copy(
                field_to_field={
                    ClapNqDocumentsConstants.ID: "document_id",
                    ClapNqDocumentsConstants.TITLE: "title",
                },
            ),
            ListFieldValues(
                fields=[ClapNqDocumentsConstants.TEXT],
                to_field="passages",
            ),
            Set(
                fields={
                    "metadata_field": "",
                }
            ),
        ],
        task="tasks.rag.corpora",
        templates=TemplatesDict(
            {
                "empty": InputOutputTemplate(
                    input_format="",
                    output_format="",
                ),
            }
        ),
    )

    # Not strict because Rouge does not return 1.0 even if predictions are equal to references, when txt is only "-'"
    # Check only one language to speed up process
    test_card(
        card,
        strict=False,
        debug=False,
        demos_taken_from="test",
        demos_pool_size=5,
    )
    add_to_catalog(card, card_name, overwrite=True)
