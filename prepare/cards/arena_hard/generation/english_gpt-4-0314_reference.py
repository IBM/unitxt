from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromHFSpace
from unitxt.operators import (
    RenameFields,
    Set,
)
from unitxt.test_utils.card import test_card

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
        RenameFields(
            field_to_field={
                "model_input": "input",
                "category": "group",
                "model_1_output": "output",
            }
        ),
        Set(
            fields={
                "type_of_input": "prompt",
                "type_of_output": "answer",
            }
        ),
    ],
    task="tasks.generation",
    templates=["templates.empty"],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=100000)
add_to_catalog(
    card,
    "cards.arena_hard.generation.english_gpt_4_0314_reference",
    overwrite=True,
)
