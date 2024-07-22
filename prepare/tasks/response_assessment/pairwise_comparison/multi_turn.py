from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "dialog_a": "List[Tuple[str, str]]",
            "dialog_b": "List[Tuple[str, str]]",
        },
        reference_fields={
            "winner": "str"
        },  # TODO: Support and change to "Literal['choice_a', 'choice_b', 'tie']"},
        metrics=["metrics.accuracy", "metrics.f1_micro", "metrics.f1_macro"],
    ),
    "tasks.response_assessment.pairwise_comparison.multi_turn",
    overwrite=True,
)
