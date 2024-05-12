from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "question": "str",
            "answer_a": "str",
            "answer_b": "str",
        },
        outputs={"winner": "Literal['choice_a', 'choice_b', 'tie']"},
        metrics=["metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparison.single_turn",
    overwrite=True,
)
