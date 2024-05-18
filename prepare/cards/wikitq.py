from unitxt.blocks import (
    AddFields,
    LoadHF,
    SerializeTableAsIndexedRowMajor,
    TaskCard,
    TruncateTableCells,
    TruncateTableRows,
)
from unitxt.catalog import add_to_catalog
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="wikitablequestions"),
    preprocess_steps=[
        "splitters.small_no_test",
        AddFields({"context_type": "table"}),
        TruncateTableCells(max_length=15, table="table", text_output="answers"),
        TruncateTableRows(field="table", rows_to_keep=50),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "context"]]),
    ],
    task="tasks.qa.with_context.extractive",
    templates="templates.qa.with_context.all",
    __description__=(
        "The WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables."
    ),
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://arxiv.org/abs/1508.00305"},
        "languages": ["english"],
        "dataset_info_tags": [
            "task_categories:question-answering",
            "annotations_creators:crowdsourced",
            "language_creators:found",
            "multilinguality:monolingual",
            "size_categories:10K<n<100K",
            "source_datasets:original",
            "language:en",
            "license:cc-by-4.0",
            "table-question-answering",
            "croissant",
            "arxiv:1508.00305",
            "region:us",
        ],
    },
)

test_card(card)
add_to_catalog(card, "cards.wikitq", overwrite=True)
