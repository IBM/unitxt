from unitxt import add_to_catalog
from unitxt.templates import DialogDictTemplate

add_to_catalog(
    DialogDictTemplate(
        field="dialog",
        input_format="{contexts}\n\n{dialog}",
        references_field="reference_answers",
    ),
    "templates.rag.response_generation.multi_turn.simple",
    overwrite=True,
)
