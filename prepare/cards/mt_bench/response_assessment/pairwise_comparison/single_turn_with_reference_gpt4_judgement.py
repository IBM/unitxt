from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Copy,
    FilterByCondition,
    MapInstanceValues,
    RenameFields,
)
from unitxt.processors import LiteralEval
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(
        path="OfirArviv/mt_bench_pairwise_comparison_gpt4_judgments", split="train"
    ),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="ne"),
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
                "reference": "reference_answer",
                "category": "group",
            }
        ),
        LiteralEval("question", to_field="question"),
        Copy(field="question/0", to_field="question"),
        LiteralEval("answer_a", to_field="answer_a"),
        Copy(field="answer_a/0", to_field="answer_a"),
        LiteralEval("answer_b", to_field="answer_b"),
        Copy(field="answer_b/0", to_field="answer_b"),
        LiteralEval("reference_answer", to_field="reference_answer"),
        Copy(field="reference_answer/0", to_field="reference_answer"),
    ],
    task="tasks.response_assessment.pairwise_comparison.single_turn_with_reference",
    templates=[
        "templates.response_assessment.pairwise_comparison.mt_bench_single_turn_with_reference_with_shuffle"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.pairwise_comparison.single_turn_with_reference_gpt4_judgement",
    overwrite=True,
)
