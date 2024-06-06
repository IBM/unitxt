from unitxt.blocks import (
    TaskCard,
)
from unitxt.catalog import add_to_catalog
from unitxt.loaders import LoadFromHFSpace
from unitxt.operators import (
    Copy,
    FilterByCondition,
    MapInstanceValues,
    RenameFields,
)
from unitxt.test_utils.card import test_card

card = TaskCard(
    loader=LoadFromHFSpace(
        space_name="lmsys/mt-bench",
        revision="a4b674c",  # Nov 4, 2023
        data_files={
            "questions": "data/mt_bench/question.jsonl",
            "model_answer": "data/mt_bench/model_answer/*.jsonl",
            "judgment": "data/mt_bench/model_judgment/gpt-4_pair.jsonl",
        },
    ),
    preprocess_steps=[
        "operators.mt_bench.pairwise_hf_space_processing_steps",
        FilterByCondition(values={"turn": 1}, condition="eq"),
        FilterByCondition(values={"reference": None}, condition="eq"),
        FilterByCondition(
            values={"winner": ["model_1", "tie", "model_2"]}, condition="in"
        ),
        MapInstanceValues(
            mappers={
                "winner": {"model_1": "choice_a", "model_2": "choice_b", "tie": "tie"}
            }
        ),
        RenameFields(
            field_to_field={
                "model_input": "question",
                "model_1_output": "answer_a",
                "model_2_output": "answer_b",
                "category": "group",
            }
        ),
        Copy(field="question/0", to_field="question"),
        Copy(field="answer_a/0", to_field="answer_a"),
        Copy(field="answer_b/0", to_field="answer_b"),
    ],
    task="tasks.response_assessment.pairwise_comparison.single_turn",
    templates=[
        "templates.response_assessment.pairwise_comparison.mt_bench_single_turn_with_shuffle"
    ],
)

test_card(card, demos_taken_from="test", strict=False, loader_limit=1000)
add_to_catalog(
    card,
    "cards.mt_bench.response_assessment.pairwise_comparison.single_turn_gpt4_judgement",
    overwrite=True,
)
