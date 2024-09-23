from unitxt.blocks import (
    SerializeTableAsIndexedRowMajor,
    Set,
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadCSV
from unitxt.processors import LiteralEval
from unitxt.splitters import SplitRandomMix
from unitxt.templates import MultiReferenceTemplate, TemplatesList
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadCSV(
        sep=",",
        files={
            "test": "/Users/shir/Downloads/simple_dataset.csv",
        },
    ),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "test[35%]", "validation": "test[10%]", "test": "test[55%]"}
        ),
        Set({"context_type": "table"}),
        LiteralEval(field="table"),
        LiteralEval(field="answers"),
        SerializeTableAsIndexedRowMajor(field_to_field=[["table", "context"]]),
    ],
    task="tasks.qa.with_context.extractive[metrics=[metrics.f1_strings, metrics.unsorted_list_exact_match]]",
    templates=TemplatesList(
        [
            MultiReferenceTemplate(
                input_format="Based on this {context_type}: {context}\nAnswer the question: {question}",
                references_field="answers",
                output_format="[[{answers}]]",
                postprocessors=[
                    # "processors.to_list_by_comma_space",
                    "processors.str_to_float_format",
                    "processors.lower_case",
                ],
            ),
        ]
    ),
)
test_card(card, strict=False)
add_to_catalog(card, "cards.simple_dataset", overwrite=True)
