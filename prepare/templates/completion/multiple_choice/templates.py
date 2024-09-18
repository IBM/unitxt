from unitxt.catalog import add_to_catalog
from unitxt.templates import MultipleChoiceTemplate, TemplatesList

add_to_catalog(
    MultipleChoiceTemplate(
        input_format="{context}",
        target_field="answer",
        choices_separator="",
        source_choice_format="{choice_text}",
        target_choice_format="{choice_text}",
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
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.completion.multiple_choice.enumerated",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Pick the best ending to the {context_type}.",
        input_format="Context: {context}...\nChoices:\n{choices}",
        target_prefix="Answer: ",
        target_field="answer",
        choices_separator="\n",
        postprocessors=["processors.first_character"],
    ),
    "templates.completion.multiple_choice.standard",
    overwrite=True,
)

add_to_catalog(
    MultipleChoiceTemplate(
        instruction="Pick the best ending to the {context_type}.",
        input_format="{context_type}: {context}\nChoices:\n{choices}",
        target_prefix="Answer: ",
        target_field="answer",
        choices_separator="\n",
        title_fields=["context_type"],
        postprocessors=["processors.first_character"],
    ),
    "templates.completion.multiple_choice.title",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.completion.multiple_choice.simple",
            "templates.completion.multiple_choice.enumerated",
            "templates.completion.multiple_choice.standard",
            "templates.completion.multiple_choice.title",
        ]
    ),
    "templates.completion.multiple_choice.all",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.completion.multiple_choice.simple",
            "templates.completion.multiple_choice.enumerated",
            "templates.completion.multiple_choice.standard",
            # "templates.completion.multiple_choice.title",
        ]
    ),
    "templates.completion.multiple_choice.simple_enumerated_standard",
    overwrite=True,
)
