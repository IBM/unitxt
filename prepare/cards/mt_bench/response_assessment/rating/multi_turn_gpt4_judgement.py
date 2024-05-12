import ast

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    Apply,
    FilterByCondition,
    InterleaveListsToDialogOperator,
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 2}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="eq"),
        RenameFields(field_to_field={"score": "rating", "category": "group"}),
        Apply("model_input", function=ast.literal_eval, to_field="model_input"),
        Apply("model_output", function=ast.literal_eval, to_field="model_output"),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="model_output",
            to_field="dialog",
        ),
    ],
    task="tasks.response_assessment.rating.multi_turn",
    templates=["templates.response_assessment.rating.mt_bench_multi_turn"],
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.rating.multi_turn_gpt4_judgement",
    overwrite=True,
)
