from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    CopyFields,
    FilterByCondition,
    RenameFields,
)
from unitxt.processors import LiteralEval
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="ne"),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "score": "rating",
                "category": "group",
                "reference": "reference_answer",
                "model_output": "answer",
            }
        ),
        LiteralEval("question", to_field="question"),
        CopyFields(field_to_field={"question/0": "question"}),
        LiteralEval("answer", to_field="answer"),
        CopyFields(field_to_field={"answer/0": "answer"}),
        LiteralEval("reference_answer", to_field="reference_answer"),
        CopyFields(field_to_field={"reference_answer/0": "reference_answer"}),
    ],
    task="tasks.response_assessment.rating.single_turn_with_reference",
    templates=[
        "templates.response_assessment.rating.mt_bench_single_turn_with_reference"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=5000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.rating.single_turn_with_reference_gpt4_judgement",
    overwrite=True,
)
