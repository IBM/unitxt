from unitxt import add_to_catalog
from unitxt.templates import MultiReferenceTemplate, TemplatesList

add_to_catalog(
    MultiReferenceTemplate(
        instruction="Make the minimal amount of changes to correct grammar and spelling errors in the following text.\n",
        input_format="Original text: {original_text}",
        references_field="corrected_texts",
        target_prefix="Corrected text: ",
    ),
    "templates.grammatical_error_correction.simple",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.grammatical_error_correction.simple",
        ]
    ),
    "templates.grammatical_error_correction.all",
    overwrite=True,
)
