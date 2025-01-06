from typing import List, Literal, Tuple

from unitxt.blocks import Task
from unitxt.catalog import add_to_catalog

add_to_catalog(
    Task(
        input_fields={
            "dialog_a": List[Tuple[str, str]],
            "dialog_b": List[Tuple[str, str]],
            "reference_dialog": List[Tuple[str, str]],
        },
        reference_fields={
            "winner": Literal["choice_a", "choice_b", "tie"],
            "classes": List[Literal["choice_a", "choice_b", "tie"]],
        },
        defaults={"classes": ["choice_a", "choice_b", "tie"]},
        metrics=["metrics.accuracy", "metrics.f1_micro", "metrics.f1_macro"],
        prediction_type=str,
    ),
    "tasks.response_assessment.pairwise_comparison.multi_turn_with_reference",
    overwrite=True,
)
