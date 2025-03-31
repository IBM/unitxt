from unitxt import add_to_catalog
from unitxt.operator import SequentialOperator
from unitxt.struct_data_operators import LoadJson
from unitxt.templates import JsonOutputTemplate

add_to_catalog(
    SequentialOperator(
        steps=[
            LoadJson(
                field="prediction",
                process_every_value=False,
            ),
        ]
    ),
    "processors.load_json_predictions",
    overwrite=True,
)

add_to_catalog(
    # For rag end-to-end tasks
    JsonOutputTemplate(
        input_format="",
        output_fields={
            "reference_answers": "answer",
            "reference_contexts": "contexts",
            "reference_context_ids": "context_ids",
        },
        wrap_with_list_fields=["reference_contexts", "reference_context_ids"],
        postprocessors=["processors.load_json_predictions"],
    ),
    "templates.rag.end_to_end.json_predictions",
    overwrite=True,
)
