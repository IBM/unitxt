from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "question": str,
            "answer_a": str,
            "answer_b": str,
        },
        outputs={
            "winner": str
        },  # TODO: Support and change to "Literal['choice_a', 'choice_b', 'tie']"
        metrics=["metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparison.single_turn",
    overwrite=True,
)
