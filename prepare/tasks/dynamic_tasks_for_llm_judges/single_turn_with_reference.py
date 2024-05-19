from unitxt.blocks import FormTask
from unitxt.catalog import add_to_catalog

add_to_catalog(
    FormTask(
        inputs={"question": "str", "answer": "str", "reference_answer": "str"},
        outputs={"score": "Any"},
        metrics=["metrics.spearman"],
    ),
    "tasks.dynamic_tasks_for_llm_judges.input_output_reference",
    overwrite=True,
)
