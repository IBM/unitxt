import ast

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Apply,
    CopyFields,
    FilterByCondition,
    MapInstanceValues,
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="OfirArviv/mt_bench_pairwise_comparison_gpt4_judgments", split="train"
    ),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="eq"),
        FilterByCondition(
            values={"winner": ["model_1", "tie", "model_2"]}, condition="in"
        ),
        MapInstanceValues(
            mappers={
                "winner": {"model_1": "choice_a", "model_2": "choice_b", "tie": "tie"}
            }
        ),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "model_1_output": "answer_a",
                "model_2_output": "answer_b",
                "category": "group",
            }
        ),
        Apply("question", function=ast.literal_eval, to_field="question"),
        CopyFields(field_to_field={"question/0": "question"}),
        Apply("answer_a", function=ast.literal_eval, to_field="answer_a"),
        CopyFields(field_to_field={"answer_a/0": "answer_a"}),
        Apply("answer_b", function=ast.literal_eval, to_field="answer_b"),
        CopyFields(field_to_field={"answer_b/0": "answer_b"}),
    ],
    task="tasks.response_assessment.pairwise_comparison.single_turn",
    templates=[
        "templates.response_assessment.pairwise_comparison.mt_bench_single_turn_with_shuffle"
    ],
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.pairwise_comparison.single_turn_gpt4_judgement",
    overwrite=True,
)
