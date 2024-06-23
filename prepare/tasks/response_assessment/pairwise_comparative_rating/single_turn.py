from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        inputs={
            "question": "str",
            "answer_a": "str",
            "answer_b": "str",
        },
        outputs={
            "winner": "str",
        },
        prediction_type="str",
        metrics=["metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparative_rating.single_turn",
    overwrite=True,
)
