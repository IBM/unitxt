from unitxt.blocks import (
    LoadHF,
    Rename,
    Task,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.operators import RemoveFields, Set
from unitxt.struct_data_operators import LoadJson
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.card import test_card
from unitxt.types import Table

card = TaskCard(
    loader=LoadHF(
        path="Multilingual-Multimodal-NLP/TableBench",
        revision="90593ad8",
        data_classification_policy=["public"],
        splits=["test"],
        filtering_lambda="lambda x: x['instruction_type'] == 'DP'",
    ),
    preprocess_steps=[
        LoadJson(field="table"),
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
