from src.unitxt import add_to_catalog
from src.unitxt.templates import MultiReferenceTemplate

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Make the minimal amount of changes to correct grammar and spelling errors in the following text.",
        input_format="Original text: {original_text}",
        references_field="corrected_texts",
        target_prefix="Corrected text: ",
    ),
    "templates.grammatical_error_correction.simple",
    overwrite=True,
)
