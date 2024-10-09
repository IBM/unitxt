from unitxt.blocks import (
    LoadHF,
    Rename,
    Task,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Apply, FilterByCondition, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card
from unitxt.types import Table

card = TaskCard(
    loader=LoadHF(
        path="Multilingual-Multimodal-NLP/TableBench",
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        # consider samples with DP(Direct Prompting) as instruction type
        FilterByCondition(values={"instruction_type": "DP"}, condition="eq"),
        # filter by question type if needed
        # FilterByCondition(values={"qtype": ["FactChecking", "NumericalReasoning"]}, condition="in"),
        # parse table json
        Apply("table", function="json.loads", to_field="table"),
        # rename table fields to match with standard table format
        Rename(
            field_to_field={"table/columns": "table/header", "table/data": "table/rows"}
        ),
        Set({"context_type": "Table"}),
        Rename(field_to_field={"table": "context", "answer": "answers"}),
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
    ),
    templates=[
        InputOutputTemplate(
            input_format="You are a table analyst. Your task is to answer questions based on the table content. {answer_formatter} \n{context_type}: {context} \nQuestion: {question}",
            target_prefix="Final Answer: ",
            output_format="{answers}",
            postprocessors=[
                "processors.to_string_stripped",
                "processors.lower_case",
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

test_card(card, strict=False, loader_limit=200)
add_to_catalog(card, "cards.tablebench", overwrite=True)
