from src.unitxt.catalog import add_to_catalog
from src.unitxt.templates import MultipleChoiceTemplate, TemplatesList

input_format = "{context}"
add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="",
        source_choice_format="{choice_text}",
        target_choice_format=" {choice_text}",
        postprocessors=["processors.first_character"],
    ),
    "templates.completion.multiple_choice.simple",
    overwrite=True,
)

# enumerated
input_format = """
Pick the best ending to the context.
Context: {context}...
Choices:
{choices}
Answer:
""".strip()  # https://rowanzellers.com/hellaswag/

add_to_catalog(
    MultipleChoiceTemplate(
        input_format=input_format,
        target_field="answer",
        choices_seperator="\n",
        add_numerals_as_field="numerals",
        postprocessors=["processors.first_character"],
    ),
    "templates.completion.multiple_choice.enumerated",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.completion.multiple_choice.simple",
            "templates.completion.multiple_choice.enumerated",
        ]
    ),
    "templates.completion.multiple_choice.all",
    overwrite=True,
)
