from src.unitxt import add_to_catalog
from src.unitxt.templates import MultipleChoiceTemplate, TemplatesList

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="For any Instruction and {input_type} given to a model, assess which {output_type} written by the model aligns most closely with the given instruction (choose from {numerals}).",
        input_format="Instruction:\n{instruction}\n{input_type}:\n{input}\nResponses:\n{choices}",
        target_prefix="{output_type}:\n",
        target_field="output_choice",
        choices_seperator="\n",
        postprocessors=["processors.to_string_stripped", "processors.first_character"],
        shuffle_choices=True,
        title_fields=["input_type", "output_type"],
    ),
    "templates.evaluation.preference.default",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.evaluation.preference.default",
        ]
    ),
    "templates.evaluation.preference.all",
    overwrite=True,
)
