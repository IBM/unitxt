from unitxt import add_to_catalog
from unitxt.templates import MultipleChoiceTemplate, TemplatesList

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Which of the {choices_text_type} is the most {required_attribute}, please respond with: {numerals}.",
        input_format="{choices_text_type}:\n{choices_texts}",
        target_prefix="Most {required_attribute}:\n",
        target_field="choice",
        choices_field="choices_texts",
        choices_seperator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
        shuffle_choices=True,
        title_fields=["choices_text_type"],
    ),
    "templates.selection.by_attribute.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.selection.by_attribute.default",
        ]
    ),
    "templates.selection.by_attribute.all",
    overwrite=True,
)
