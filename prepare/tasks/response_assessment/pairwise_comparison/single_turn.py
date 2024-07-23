from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "question": "str",
            "answer_a": "str",
            "answer_b": "str",
        },
        reference_fields={
            "winner": "str"
        },  # TODO: Support and change to "Literal['choice_a', 'choice_b', 'tie']"
        metrics=["metrics.accuracy", "metrics.f1_micro", "metrics.f1_macro"],
    ),
    "tasks.response_assessment.pairwise_comparison.single_turn",
    overwrite=True,
)
