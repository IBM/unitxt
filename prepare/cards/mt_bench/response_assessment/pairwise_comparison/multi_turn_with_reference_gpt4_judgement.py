from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    FilterByCondition,
    InterleaveListsToDialogOperator,
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
        FilterByCondition(values={"turn": 2}, condition="eq"),
        FilterByCondition(values={"reference": "[]"}, condition="ne"),
        FilterByCondition(
            values={"winner": ["model_1", "tie", "model_2"]}, condition="in"
        ),
        MapInstanceValues(
            mappers={
                "winner": {"model_1": "choice_a", "model_2": "choice_b", "tie": "tie"}
            }
        ),
        RenameFields(field_to_field={"category": "group"}),
        LiteralEval("model_input", to_field="model_input"),
        LiteralEval("model_1_output", to_field="model_1_output"),
        LiteralEval("model_2_output", to_field="model_2_output"),
        LiteralEval("reference", to_field="reference"),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="model_1_output",
            to_field="dialog_a",
        ),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="model_2_output",
            to_field="dialog_b",
        ),
        InterleaveListsToDialogOperator(
            user_turns_field="model_input",
            assistant_turns_field="reference",
            to_field="reference_dialog",
        ),
    ],
    task="tasks.response_assessment.pairwise_comparison.multi_turn_with_reference",
    templates=[
        "templates.response_assessment.pairwise_comparison.mt_bench_multi_turn_with_reference_with_shuffle"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=2000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.pairwise_comparison.multi_turn_with_reference_gpt4_judgement",
    overwrite=True,
)
