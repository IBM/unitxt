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
    InterleaveListsToDialogOperator,
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

"""
The original MtBench data with GPT4 score. This is the single score variation.
There are 4 subset (and as such, cards):
- mt_bench_model_rating_single_turn_gpt4_judgement
- mt_bench_model_rating_with_reference_single_turn_gpt4_judgement
- mt_bench_model_rating_multi_turn_gpt4_judgement
- mt_bench_model_rating_with_reference_multi_turn_gpt4_judgement
"""


# region builders methods
def create_mt_bench_model_rating_single_turn_gpt4_judgement():
    card = TaskCard(
        loader=LoadHF(
            path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"
        ),
        preprocess_steps=[
            RenameSplits({"train": "test"}),
            FilterByCondition(values={"turn": 1}, condition="eq"),
            FilterByCondition(values={"reference": "[]"}, condition="eq"),
            RenameFields(
                field_to_field={
                    "model_input": "question",
                    "score": "rating",
                    "category": "group",
                    "model_output": "model_answer",
                }
            ),
            Apply("question", function=ast.literal_eval, to_field="question"),
            CopyFields(field_to_field={"question/0": "question"}),
            Apply("model_answer", function=ast.literal_eval, to_field="model_answer"),
            CopyFields(field_to_field={"model_answer/0": "model_answer"}),
        ],
        task="tasks.model_response_assessment.model_rating_single_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_rating_single_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_rating_single_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_rating_with_reference_single_turn_gpt4_judgement():
    card = TaskCard(
        loader=LoadHF(
            path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"
        ),
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
                    "model_output": "model_answer",
                }
            ),
            Apply("question", function=ast.literal_eval, to_field="question"),
            CopyFields(field_to_field={"question/0": "question"}),
            Apply("model_answer", function=ast.literal_eval, to_field="model_answer"),
            CopyFields(field_to_field={"model_answer/0": "model_answer"}),
            Apply(
                "reference_answer",
                function=ast.literal_eval,
                to_field="reference_answer",
            ),
            CopyFields(field_to_field={"reference_answer/0": "reference_answer"}),
        ],
        task="tasks.model_response_assessment.model_rating_with_reference_single_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_rating_with_reference_single_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_rating_with_reference_single_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_rating_multi_turn_gpt4_judgement():
    card = TaskCard(
        loader=LoadHF(
            path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"
        ),
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
        task="tasks.model_response_assessment.model_rating_multi_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_rating_multi_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_rating_multi_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_rating_with_reference_multi_turn_gpt4_judgement():
    card = TaskCard(
        loader=LoadHF(
            path="OfirArviv/mt_bench_single_score_gpt4_judgement", split="train"
        ),
        preprocess_steps=[
            RenameSplits({"train": "test"}),
            FilterByCondition(values={"turn": 2}, condition="eq"),
            FilterByCondition(values={"reference": "[]"}, condition="ne"),
            RenameFields(field_to_field={"score": "rating", "category": "group"}),
            Apply("model_input", function=ast.literal_eval, to_field="model_input"),
            Apply("model_output", function=ast.literal_eval, to_field="model_output"),
            Apply("reference", function=ast.literal_eval, to_field="reference"),
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
        task="tasks.model_response_assessment.model_rating_with_reference_multi_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_rating_with_reference_multi_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_rating_with_reference_multi_turn_gpt4_judgement",
        overwrite=True,
    )


# endregion

create_mt_bench_model_rating_single_turn_gpt4_judgement()
create_mt_bench_model_rating_with_reference_single_turn_gpt4_judgement()
create_mt_bench_model_rating_multi_turn_gpt4_judgement()
create_mt_bench_model_rating_with_reference_multi_turn_gpt4_judgement()
