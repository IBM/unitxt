import math
import random
from dataclasses import dataclass

import pandas as pd
from unitxt import add_to_catalog
from unitxt.blocks import TaskCard, TemplatesDict
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, ListFieldValues, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card

from prepare.cards.rag.end_to_end.clap_nq_benchmark import (
    ClapNqBenchmarkConstants,
)
from prepare.tasks.rag.rag_task import (
    TaskRagCorporaConstants,
    TaskRagCorporaInputConstants,
)


@dataclass(frozen=True)
class ClapNqDocumentsConstants:
    # Raw_data
    RAW_FILE_URL: str = "https://media.githubusercontent.com/media/primeqa/clapnq/main/retrieval/passages.tsv"

    # Fields
    ID: str = "id"
    TEXT: str = "text"
    TITLE: str = "title"

    ARTIFACT_NAME: str = "cards.rag.documents.clap_nq.en"


SUBSET_SIZE = 200
RANDOM_SEED = 43
NEGATIVE_DOCUMENTS_FACTOR = 9
# We want to create a subset of ClapNq - based on a subset of200 questions from the benchmark.
# 1 - first we read the benchmark:
benchmark_train_df = pd.read_csv(ClapNqBenchmarkConstants.TRAIN_RAW_FILE_URL, sep="\t")
benchmark_test_df = pd.read_csv(ClapNqBenchmarkConstants.TEST_RAW_FILE_URL, sep="\t")

# 2 - We compute the ration between train and test - we want to keep a similar one:
ratio_train_test = len(benchmark_train_df) / len(benchmark_test_df)

# 3 - We compute the size of train subset and test subset
train_size = math.floor(SUBSET_SIZE / ratio_train_test)
test_size = SUBSET_SIZE - train_size

# 4 - We extract the subset of the rows
benchmark_train_df_subset = benchmark_train_df.sample(
    n=train_size, random_state=RANDOM_SEED
)
benchmark_test_df_subset = benchmark_test_df.sample(
    n=test_size, random_state=RANDOM_SEED
)

# 5 - We extract the relevant gt_context_ids
gt_doc_ids = set(benchmark_train_df_subset["doc-id-list"].tolist()) | set(
    benchmark_test_df_subset["doc-id-list"].tolist()
)

# 6 - We take the whole doc_ids in the corpus:
documents_df = pd.read_csv(ClapNqDocumentsConstants.RAW_FILE_URL, sep="\t")
total_docs_ids = set(documents_df["id"].tolist())

# 7 - We remove the gt_doc-ids
other_docs_ids = list(total_docs_ids - gt_doc_ids)

# 8 - We sort and shuffle the other_doc_ids for reproducibility
other_docs_ids.sort()
random.seed(RANDOM_SEED)
random.shuffle(other_docs_ids)

# 9 - We select the number of negative examples:
other_docs_ids = other_docs_ids[len(gt_doc_ids) * NEGATIVE_DOCUMENTS_FACTOR]

# 10 - We create the set of doc-ids we want to keep
selected_doc_ids = other_docs_ids | gt_doc_ids

# 11 - We create the right documents_df
subset_doc_df = documents_df[documents_df["id"].isin(selected_doc_ids)]
subset_doc_df.reset_index(drop=True)


card = TaskCard(
    loader=LoadCSV(
        sep="\t",
        files={"test": ClapNqDocumentsConstants.RAW_FILE_URL},
    ),
    preprocess_steps=[
        Copy(
            field_to_field={
                ClapNqDocumentsConstants.ID: TaskRagCorporaInputConstants.DOCUMENT_ID,
                ClapNqDocumentsConstants.TITLE: TaskRagCorporaInputConstants.TITLE,
            },
        ),
        ListFieldValues(
            fields=[ClapNqDocumentsConstants.TEXT],
            to_field=TaskRagCorporaInputConstants.PASSAGES,
        ),
        Set(
            fields={
                TaskRagCorporaInputConstants.METADATA_FIELD: "",
            }
        ),
    ],
    task=TaskRagCorporaConstants.TASKS_RAG_CORPORA,
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

add_to_catalog(card, ClapNqDocumentsConstants.ARTIFACT_NAME, overwrite=True)
