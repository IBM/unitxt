from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    FilterByCondition,
    InterleaveListsToDialogOperator,
    RenameFields,
)
from unitxt.processors import LiteralEval
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadHF(path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"),
    preprocess_steps=[
        RenameSplits({"train": "test"}),
        FilterByCondition(values={"turn": 2}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="ne"),
        RenameFields(field_to_field={"score": "rating", "category": "group"}),
        LiteralEval("model_input", to_field="model_input"),
        LiteralEval("model_output", to_field="model_output"),
        LiteralEval("reference", to_field="reference"),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="model_output",
            to_field="dialog",
        ),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="reference",
            to_field="reference_dialog",
        ),
    ],
    task="tasks.response_assessment.rating.multi_turn_with_reference",
    templates=[
        "templates.response_assessment.rating.mt_bench_multi_turn_with_reference"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=2000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.rating.multi_turn_with_reference_gpt4_judgement",
    overwrite=True,
)
