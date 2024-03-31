from src.unitxt import add_to_catalog
from src.unitxt.task import FormTask

add_to_catalog(
    FormTask(
        inputs=["original_text"],
        outputs=["corrected_texts"],
        metrics=[
            "metrics.char_edit_dist_accuracy",
            "metrics.rouge",
            "metrics.char_edit_distance[reference_field=original_text]",
        ],
    ),
    "tasks.grammatical_error_correction",
    overwrite=True,
)
