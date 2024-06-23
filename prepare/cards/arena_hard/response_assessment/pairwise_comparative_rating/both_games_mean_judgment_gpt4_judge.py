from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromHFSpace
from unitxt.operators import (
    ExecuteExpression,
    MapInstanceValues,
    RenameFields,
)
from unitxt.test_utils.card import test_card

score_mapper = {"A=B": 0, "A>B": 1, "A>>B": 3, "B>A": -1, "B>>A": -3}

score_mapper_reversed = {k: -1 * v for k, v in score_mapper.items()}

card = TaskCard(
    loader=LoadFromHFSpace(
        space_name="lmsys/arena-hard-browser",
        revision="03b91ca",  # May 26, 2024
        data_files={
            "questions": "data/arena-hard-v0.1/question.jsonl",
            "model_answer": "data/arena-hard-v0.1/model_answer/*.jsonl",
            "judgment": "data/arena-hard-v0.1/model_judgment/gpt-4-1106-preview/*.jsonl",
        },
    ),
    data_classification_policy=["public"],
    preprocess_steps=[
        "operators.arena_hard_hf_space_processing_steps",
        MapInstanceValues(
            {
                "score_model_1_ordered_first": score_mapper,
                "score_model_2_ordered_first": score_mapper_reversed,
            }
        ),
        ExecuteExpression(
            to_field="answer_a_preference",
            expression="int(round((score_model_1_ordered_first+score_model_2_ordered_first)/2))",
        ),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "model_1_output": "answer_a",
                "model_2_output": "answer_b",
                "category": "group",
                "model_1": "model_a",
                "model_2": "model_b",
            }
        ),
    ],
    task="tasks.response_assessment.pairwise_comparative_rating.single_turn",
    templates=[
        "templates.response_assessment.pairwise_comparative_rating.arena_hard",
        "templates.response_assessment.pairwise_comparative_rating.arena_hard_with_shuffling",
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=100000)
add_to_catalog(
    card,
    "cards.arena_hard.response_assessment.pairwise_comparative_rating.both_games_mean_judgment_gpt4_judge",
    overwrite=True,
)
