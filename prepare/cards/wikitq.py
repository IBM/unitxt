from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadCSV
from unitxt.operators import Copy, FilterByCondition, ReadFile, Rename, Set
from unitxt.string_operators import FormatText, Split
from unitxt.struct_data_operators import GetNumOfTableCells, ParseCSV
from unitxt.templates import MultiReferenceTemplate
from unitxt.test_utils.card import test_card

base_url = "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/data"
table_url_format = "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/refs/heads/master/{context}"

card = TaskCard(
    loader=LoadCSV(
        files={
            "train": f"{base_url}/random-split-1-train.tsv",
            "validation": f"{base_url}/random-split-1-dev.tsv",
            "test": f"{base_url}/pristine-unseen-tables.tsv",
        },
        sep="\t",
        # column_names=["id", "question", "table_name", "answer"],
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        Rename(field="utterance", to_field="question"),
        Split(field="targetValue", to_field="answers", by="|"),
        Set({"context_type": "table"}),
        FormatText(text=table_url_format, to_field="table_url"),
        ReadFile(field="table_url", to_field="table_content"),
        ParseCSV(field="table_content", to_field="table", separator="\t"),
        GetNumOfTableCells(field="table", to_field="table_cell_size"),
        FilterByCondition(values={"table_cell_size": 200}, condition="le"),
        Copy(field="table", to_field="context"),
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
