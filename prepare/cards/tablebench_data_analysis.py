from unitxt.blocks import (
    LoadHF,
    Rename,
    Task,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Apply, FilterByCondition, RemoveFields, Set
from unitxt.splitters import SplitRandomMix
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card
from unitxt.types import Table

card = TaskCard(
    loader=LoadHF(
        path="Multilingual-Multimodal-NLP/TableBench",
        revision="90593ad8af90f027f6f478b8c4c1981d9f073a83", # pragma: allowlist secret
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        SplitRandomMix(
            mix={
                "train": "test[20%]",
                "validation": "test[20%]",
                "test": "test[60%]",
            }
        ),
        FilterByCondition(values={"instruction_type": "DP"}, condition="eq"),
        FilterByCondition(
            values={"qtype": ["DataAnalysis"]}, condition="in"
        ),  # filter by question type if needed
        # consider samples with DP(Direct Prompting) as instruction type
        Apply("table", function="json.loads", to_field="table"),  # parse table json
        # rename table fields to match with standard table format
        Rename(
            field_to_field={"table/columns": "table/header", "table/data": "table/rows"}
        ),
        Set({"context_type": "Table"}),
        Rename(field_to_field={"table": "context", "answer": "answers"}),
        RemoveFields(fields=["instruction"]),


    ],
    task=Task(
        input_fields={
            "context": Table,
            "context_type": str,
            "question": str,
            "answer_formatter": str,
        },
        reference_fields={"answers": str},
        prediction_type=str,
        metrics=["metrics.rouge"],
        augmentable_inputs=["context", "question"],
    ),
    templates=[
        InputOutputTemplate(
            instruction="You are a table analyst. Your task is to answer questions based on the table content. {answer_formatter}"
            + "\nOutput only the final answer without any explanations, extra information, or introductory text."
            + "\nHere are some input-output examples. Read the examples carefully to figure out the mapping. The output of the last example is not given, and your job is to figure out what it is.",
            input_format="{context_type}: {context} \nQuestion: {question}",
            target_prefix="Final Answer: ",
            output_format="{answers}",
            postprocessors=[
                "processors.take_first_non_empty_line",
                "processors.lower_case",
                "processors.remove_punctuations",
                "processors.remove_articles",
                "processors.fix_whitespace",
            ],
        ),
    ],
    __description__=(
        "This TableBench dataset is a Comprehensive and Complex Benchmark for Table Question Answering. For more details, refer to https://tablebench.github.io/"
    ),
    __tags__={
        "modality": "table",
        "urls": {"arxiv": "https://www.arxiv.org/pdf/2408.09174"},
        "languages": ["english"],
    },
)

test_card(card, strict=False, loader_limit=200, demos_pool_size=-1, num_demos=1)
add_to_catalog(card, "cards.tablebench_data_analysis", overwrite=True)
