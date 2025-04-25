from unitxt.blocks import (
    LoadHF,
    Rename,
    Task,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import Apply, FilterByCondition, RemoveFields, Set
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card
from unitxt.types import Table

card = TaskCard(
    loader=LoadHF(
        path="Multilingual-Multimodal-NLP/TableBench",
        revision="90593ad8af90f027f6f478b8c4c1981d9f073a83", # pragma: allowlist secret
        data_classification_policy=["public"],
        splits=["test"],
    ),
    preprocess_steps=[
        # consider samples with DP(Direct Prompting) as instruction type
        FilterByCondition(values={"instruction_type": "DP"}, condition="eq"),
        # FilterByCondition(values={"qtype": ["FactChecking", "NumericalReasoning"]}, condition="in"), # filter by question type if needed
        Apply("table", function="json.loads", to_field="table"),  # parse table json
        # rename table fields to match with standard table format
        Rename(
            field_to_field={"table/columns": "table/header", "table/data": "table/rows"}
        ),
        Set({"context_type": "Table"}),

        Rename(field_to_field={"table": "context", "answer": "answers"}),
        RemoveFields(fields=["instruction"])

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

test_card(card, strict=False, loader_limit=200)
add_to_catalog(card, "cards.tablebench", overwrite=True)
