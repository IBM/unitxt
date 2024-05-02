from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"question": "str", "model_output": "str"},
        outputs={"rating_label": "int"},
        metrics=["metrics.spearman"],
    ),
    "tasks.rag.model_response_assessment",
    overwrite=True,
)
