from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs=["question", "model_output"],
        outputs=["rating_label"],
        metrics=["metrics.spearman"],
    ),
    "tasks.llm_as_judge.rag.model_response_assessment",
    overwrite=True,
)
