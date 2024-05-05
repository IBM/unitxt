from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"dialog": "List[Tuple[str, str]]"},
        outputs={"rating_label": "float"},
        metrics=["metrics.spearman"],
    ),
    "tasks.rag.model_response_assessment_chat",
    overwrite=True,
)
