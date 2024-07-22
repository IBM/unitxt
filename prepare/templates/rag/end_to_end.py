from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    # For rag end-to-end tasks
    InputOutputTemplate(
        input_format="",
        output_format='{{"answer": "{reference_answers}", "contexts" : ["{reference_contexts}"],  "context_ids" : ["{reference_context_ids}"]}}',
        postprocessors=["processors.load_json_predictions"],
    ),
    "templates.rag.end_to_end.json_predictions",
    overwrite=True,
)
