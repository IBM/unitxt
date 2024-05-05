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
    loader=LoadHF(path="Asaf-Yehudai/mt_bench_gpt_4_judge_single_score", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 2}, condition="eq"),
        RenameFields(field_to_field={"score": "rating_label"}),
        Apply("model_input", function=ast.literal_eval, to_field="model_input"),
        Apply("model_output", function=ast.literal_eval, to_field="model_output"),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="model_output",
            to_field="dialog",
        ),
    ],
    task="tasks.model_response_assessment.absolute_score_dialog",
    templates=["templates.model_response_assessment.mt_bench_absolute_score_dialog"],
)

test_card(card, demos_taken_from="test", strict=False)
add_to_catalog(
    card,
    "cards.model_response_assessment.mt_bench_absolute_score_2_turns_gpt4_judge",
    overwrite=True,
)
