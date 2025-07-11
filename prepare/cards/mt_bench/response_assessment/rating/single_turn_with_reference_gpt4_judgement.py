from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromHFSpace
from unitxt.operators import Copy, Fillna, FilterByCondition, Rename
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadFromHFSpace(
        space_name="lmsys/mt-bench",
        revision="a4b674c",  # Nov 4, 2023
        data_files={
            "questions": "data/mt_bench/question.jsonl",
            "model_answer": "data/mt_bench/model_answer/*.jsonl",
            "judgment": "data/mt_bench/model_judgment/gpt-4_single.jsonl",
        },
        data_classification_policy=["public"],
    ),
    preprocess_steps=[
        "operators.mt_bench.rating_hf_space_processing_steps",
        FilterByCondition(values={"turn": 1}, condition="eq"),
        Fillna(field="reference", value=None),
        FilterByCondition(values={"reference": None}, condition="ne"),
        Rename(
            field_to_field={
                "model_input": "question",
                "score": "rating",
                "category": "group",
                "reference": "reference_answer",
                "model_output": "answer",
            }
        ),
        Copy(field="question/0", to_field="question"),
        Copy(field="answer/0", to_field="answer"),
        Copy(field="reference_answer/0", to_field="reference_answer"),
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
