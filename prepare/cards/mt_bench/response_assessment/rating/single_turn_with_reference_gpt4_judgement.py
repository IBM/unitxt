from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromHFSpace
from unitxt.operators import (
    CopyFields,
    FilterByCondition,
    RenameFields,
)
from unitxt.test_utils.card import test_card

from prepare.cards.mt_bench.common import mt_bench_rating_hf_space_processing_steps

card = TaskCard(
    loader=LoadFromHFSpace(
        space_name="lmsys/mt-bench",
        revision="a4b674c",  # Nov 4, 2023
        data_files={
            "questions": "data/mt_bench/question.jsonl",
            "model_answer": "data/mt_bench/model_answer/*.jsonl",
            "judgment": "data/mt_bench/model_judgment/gpt-4_single.jsonl",
        },
    ),
    preprocess_steps=[
        *mt_bench_rating_hf_space_processing_steps,
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": None}, condition="ne"),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "score": "rating",
                "category": "group",
                "reference": "reference_answer",
                "model_output": "answer",
            }
        ),
        CopyFields(field_to_field={"question/0": "question"}),
        CopyFields(field_to_field={"answer/0": "answer"}),
        CopyFields(field_to_field={"reference_answer/0": "reference_answer"}),
    ],
    task="tasks.response_assessment.rating.single_turn_with_reference",
    templates=[
        "templates.response_assessment.rating.mt_bench_single_turn_with_reference"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.rating.single_turn_with_reference_gpt4_judgement",
    overwrite=True,
)
