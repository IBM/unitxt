from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={
            "dialog_a": "List[Tuple[str, str]]",
            "dialog_b": "List[Tuple[str, str]]",
            "reference_dialog": "List[Tuple[str, str]]",
        },
        outputs={"winner": "Literal['choice_a', 'choice_b', 'tie']"},
        metrics=["metrics.accuracy"],
    ),
    "tasks.response_assessment.pairwise_comparison.multi_turn_with_reference",
    overwrite=True,
)
