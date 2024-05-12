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
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="eq"),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "score": "rating",
                "category": "group",
                "model_output": "answer",
            }
        ),
        Apply("question", function=ast.literal_eval, to_field="question"),
        CopyFields(field_to_field={"question/0": "question"}),
        Apply("answer", function=ast.literal_eval, to_field="answer"),
        CopyFields(field_to_field={"answer/0": "answer"}),
    ],
    task="tasks.response_assessment.rating.single_turn",
    templates=["templates.response_assessment.rating.mt_bench_multi_turn"],
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.rating.single_turn_gpt4_judgement",
    overwrite=True,
)
