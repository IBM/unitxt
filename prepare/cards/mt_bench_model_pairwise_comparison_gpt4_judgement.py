import ast

from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadHF
from unitxt.operators import (
    AddFields,
    Apply,
    CopyFields,
    FilterByCondition,
    InterleaveListsToDialogOperator,
    MapInstanceValues,
    RenameFields,
)
from unitxt.splitters import RenameSplits
from unitxt.test_utils.card import test_card

"""
The original MtBench data with GPT4 score. This is the pairwise comparison variation.
There are 4 subset (and as such, cards):
- mt_bench_model_pairwise_comparison_single_turn_gpt4_judgement
- mt_bench_model_pairwise_comparison_with_reference_single_turn_gpt4_judgement
- mt_bench_model_pairwise_comparison_multi_turn_gpt4_judgement
- mt_bench_model_pairwise_comparison_with_reference_multi_turn_gpt4_judgement
"""


# region builders methods


def create_mt_bench_model_pairwise_comparison_single_turn_gpt4_judgement():
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
                mappers={"winner": {"model_1": "A", "model_2": "B", "tie": "C"}}
            ),
            AddFields(
                fields={"model_a_label": "A", "model_b_label": "B", "tie_label": "C"}
            ),
            RenameFields(
                field_to_field={
                    "model_input": "question",
                    "model_1_output": "model_a_answer",
                    "model_2_output": "model_b_answer",
                    "category": "group",
                }
            ),
            Apply("question", function=ast.literal_eval, to_field="question"),
            CopyFields(field_to_field={"question/0": "question"}),
            Apply(
                "model_a_answer", function=ast.literal_eval, to_field="model_a_answer"
            ),
            CopyFields(field_to_field={"model_a_answer/0": "model_a_answer"}),
            Apply(
                "model_b_answer", function=ast.literal_eval, to_field="model_b_answer"
            ),
            CopyFields(field_to_field={"model_b_answer/0": "model_b_answer"}),
        ],
        task="tasks.model_response_assessment.model_pairwise_comparison_single_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_pairwise_comparison_single_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_pairwise_comparison_single_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_pairwise_comparison_with_reference_single_turn_gpt4_judgement():
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
                mappers={"winner": {"model_1": "A", "model_2": "B", "tie": "C"}}
            ),
            AddFields(
                fields={"model_a_label": "A", "model_b_label": "B", "tie_label": "C"}
            ),
            RenameFields(
                field_to_field={
                    "model_input": "question",
                    "model_1_output": "model_a_answer",
                    "model_2_output": "model_b_answer",
                    "reference": "reference_answer",
                    "category": "group",
                }
            ),
            Apply("question", function=ast.literal_eval, to_field="question"),
            CopyFields(field_to_field={"question/0": "question"}),
            Apply(
                "model_a_answer", function=ast.literal_eval, to_field="model_a_answer"
            ),
            CopyFields(field_to_field={"model_a_answer/0": "model_a_answer"}),
            Apply(
                "model_b_answer", function=ast.literal_eval, to_field="model_b_answer"
            ),
            CopyFields(field_to_field={"model_b_answer/0": "model_b_answer"}),
            Apply(
                "reference_answer",
                function=ast.literal_eval,
                to_field="reference_answer",
            ),
            CopyFields(field_to_field={"reference_answer/0": "reference_answer"}),
        ],
        task="tasks.model_response_assessment.model_pairwise_comparison_with_reference_single_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_pairwise_comparison_with_reference_single_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_pairwise_comparison_with_reference_single_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_pairwise_comparison_multi_turn_gpt4_judgement():
    card = TaskCard(
        loader=LoadHF(
            path="OfirArviv/mt_bench_pairwise_comparison_gpt4_judgments", split="train"
        ),
        preprocess_steps=[
            RenameSplits({"train": "test"}),
            FilterByCondition(values={"turn": 2}, condition="eq"),
            FilterByCondition(values={"reference": "[]"}, condition="eq"),
            FilterByCondition(
                values={"winner": ["model_1", "tie", "model_2"]}, condition="in"
            ),
            MapInstanceValues(
                mappers={"winner": {"model_1": "A", "model_2": "B", "tie": "C"}}
            ),
            AddFields(
                fields={"model_a_label": "A", "model_b_label": "B", "tie_label": "C"}
            ),
            RenameFields(
                field_to_field={
                    "category": "group",
                }
            ),
            Apply("model_input", function=ast.literal_eval, to_field="model_input"),
            Apply(
                "model_1_output", function=ast.literal_eval, to_field="model_1_output"
            ),
            Apply(
                "model_2_output", function=ast.literal_eval, to_field="model_2_output"
            ),
            InterleaveListsToDialogOperator(
                user_turns_field="model_input",
                assistant_turns_field="model_1_output",
                to_field="model_a_dialog",
            ),
            InterleaveListsToDialogOperator(
                user_turns_field="model_input",
                assistant_turns_field="model_2_output",
                to_field="model_b_dialog",
            ),
        ],
        task="tasks.model_response_assessment.model_pairwise_comparison_multi_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_pairwise_comparison_multi_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_pairwise_comparison_multi_turn_gpt4_judgement",
        overwrite=True,
    )


def create_mt_bench_model_pairwise_comparison_with_reference_multi_turn_gpt4_judgement():
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
                mappers={"winner": {"model_1": "A", "model_2": "B", "tie": "C"}}
            ),
            AddFields(
                fields={"model_a_label": "A", "model_b_label": "B", "tie_label": "C"}
            ),
            RenameFields(field_to_field={"category": "group"}),
            Apply("model_input", function=ast.literal_eval, to_field="model_input"),
            Apply(
                "model_1_output", function=ast.literal_eval, to_field="model_1_output"
            ),
            Apply(
                "model_2_output", function=ast.literal_eval, to_field="model_2_output"
            ),
            Apply("reference", function=ast.literal_eval, to_field="reference"),
            InterleaveListsToDialogOperator(
                user_turns_field="model_input",
                assistant_turns_field="model_1_output",
                to_field="model_a_dialog",
            ),
            InterleaveListsToDialogOperator(
                user_turns_field="model_input",
                assistant_turns_field="model_2_output",
                to_field="model_b_dialog",
            ),
            InterleaveListsToDialogOperator(
                user_turns_field="model_input",
                assistant_turns_field="reference",
                to_field="reference_dialog",
            ),
        ],
        task="tasks.model_response_assessment.model_pairwise_comparison_with_reference_multi_turn",
        templates=[
            "templates.model_response_assessment.mt_bench_model_pairwise_comparison_with_reference_multi_turn"
        ],
    )

    test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
    add_to_catalog(
        card,
        "cards.model_response_assessment.mt_bench_model_pairwise_comparison_with_reference_multi_turn_gpt4_judgement",
        overwrite=True,
    )


# endregion

create_mt_bench_model_pairwise_comparison_single_turn_gpt4_judgement()
create_mt_bench_model_pairwise_comparison_with_reference_single_turn_gpt4_judgement()
create_mt_bench_model_pairwise_comparison_multi_turn_gpt4_judgement()
create_mt_bench_model_pairwise_comparison_with_reference_multi_turn_gpt4_judgement()
