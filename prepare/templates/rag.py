from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

from prepare.tasks.rag.rag_task import TaskRagEndToEndConstants

add_to_catalog(
    # For rag end-to-end tasks
    InputOutputTemplate(
        input_format="",
        output_format='{{"answer": "{reference_answers}", "contexts" : ["{reference_contexts}"],  "context_ids" : ["{reference_context_ids}"]}}',
        postprocessors=["processors.load_json_predictions"],
    ),
    f"{TaskRagEndToEndConstants.TEMPLATE_RAG_END_TO_END_JSON_PREDICTIONS}",
)
