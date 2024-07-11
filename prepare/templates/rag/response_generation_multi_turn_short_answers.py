from unitxt import add_to_catalog
from unitxt.templates import DialogDictTemplate

add_to_catalog(
    DialogDictTemplate(
        field="dialog",
        input_format="{contexts}\n\n{dialog}",
        references_field="reference_answers",
        postprocessors=[
            "processors.take_first_non_empty_line",
            "processors.to_list_by_comma",
        ],
    ),
    "templates.rag.response_generation.multi_turn.simple_short_answers",
    overwrite=True,
)
