from typing import List

from unitxt.blocks import (
    InputOutputTemplate,
    LoadHF,
    Task,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.splitters import SplitRandomMix
from unitxt.test_utils.card import test_card
from unitxt.types import Table

card = TaskCard(
    loader=LoadHF(
        path="ibm/turl_table_col_type",
        data_classification_policy=["public", "proprietary"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            mix={
                "train": "train[50%]",
                "validation": "train[50%]",
                "test": "test+validation",
            }
        ),
    ],
    task=Task(
        input_fields={
            "page_title": str,
            "section_title": str,
            "table_caption": str,
            "table": Table,
            "vocab": List[str],
            "colname": str,
        },
        reference_fields={"annotations": List[str]},
        prediction_type="List[str]",
        metrics=[
            "metrics.f1_micro_multi_label",
            "metrics.accuracy",
            "metrics.f1_macro_multi_label",
        ],
        augmentable_inputs=[
            "page_title",
            "section_title",
            "table_caption",
            "table",
            "colname",
            "vocab",
        ],
    ),
    templates=[
        InputOutputTemplate(
            instruction="Please answer based on the following Candidate Types only: \n{vocab}",
            input_format="""
                    This is a column type annotation task. The goal of this task is to choose the correct types for one selected column of the given input table from the given candidate types. The Wikipedia page, section and table caption (if any) provide important information for choosing the correct column types. \nPage Title: {page_title} \nSection Title: {section_title} \nTable caption: {table_caption} \nTable: \n{table} \nSelected Column: {colname} \nOutput only the correct column types for this column (column name: {colname}) from the candidate types.
                """.strip(),
            output_format="{annotations}",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.lower_case",
                "processors.to_list_by_comma",
            ],
        ),
    ],
    __description__=(
        "This TURL dataset is a large-scale dataset based on WikiTables corpus for the task of column type annotation. Given a table T and a set of semantic types L, the task is to annotate a column in T with l ∈ L so that all entities in the column have type l. Note that a column can have multiple types. See the full description on the dataset page: https://github.com/sunlab-osu/TURL"
    ),
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://arxiv.org/pdf/2006.14806"},
        "languages": ["english"],
    },
)

test_card(card, num_demos=2, demos_pool_size=10)
add_to_catalog(card, "cards.turl_col_type", overwrite=True)
