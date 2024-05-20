from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "dialog_a": "List[Tuple[str, str]]",
            "dialog_b": "List[Tuple[str, str]]",
        },
        outputs={
            "winner": "str"
        },  # TODO: Support and change to "Literal['choice_a', 'choice_b', 'tie']"},
        metrics=["metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparison.multi_turn",
    overwrite=True,
)
