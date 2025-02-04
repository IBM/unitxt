from unitxt.blocks import (
    LoadHF,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Copy, FilterByCondition, Set
from unitxt.struct_data_operators import GetNumOfTableCells
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card

card = TaskCard(
    # Adjust the num_proc value according to the number of CPU cores available for faster loading
    loader=LoadHF(
        path="wikitablequestions",
        data_classification_policy=["public"],
        num_proc=10,
        all_splits=["train", "test", "validation"],
    ),
    preprocess_steps=[
        Set({"context_type": "table"}),
        GetNumOfTableCells(field="table", to_field="table_cell_size"),
        FilterByCondition(
            values={"table_cell_size": 200}, condition="le"
        ),  # filter out tables with more than 200 cells
        Copy(field="table", to_field="context"),
        # TruncateTableRows(field="table", to_field="context"),
    ],
    task="tasks.qa.extractive[metrics=[metrics.f1_strings, metrics.unsorted_list_exact_match]]",
    templates=[
        MultiReferenceTemplate(
            instruction="Answer the question based on the provided table. "
            "Extract and output only the final answer—the exact phrase or data from the table that directly answers the question. "
            "Do not include any alterations, explanations, or introductory text."
            "\nHere are some input-output examples. Read the examples carefully to figure out the mapping. "
            "The output of the last example is not given, and your job is to figure out what it is.",
            input_format="\nQuestion: {question}" "\nTable: {context}" "\nAnswer: ",
            references_field="answers",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.to_list_by_comma_space",
                "processors.str_to_float_format",
            ],
        ),
    ],
    __description__=(
        "This WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables… See the full description on the dataset page: https://huggingface.co/datasets/wikitablequestions"
    ),
    __tags__={
        "annotations_creators": "crowdsourced",
        "arxiv": "1508.00305",
        "flags": ["table-question-answering"],
        "language": "en",
        "language_creators": "found",
        "license": "cc-by-4.0",
        "multilinguality": "monolingual",
        "region": "us",
        "size_categories": "10K<n<100K",
        "source_datasets": "original",
        "task_categories": "question-answering",
    },
)

test_card(card, strict=False, num_demos=2, demos_pool_size=5)
add_to_catalog(card, "cards.wikitq", overwrite=True)
