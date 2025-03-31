from unitxt import add_to_catalog
from unitxt.templates import InputOutputTemplate, TemplatesList

add_to_catalog(
    InputOutputTemplate(
        instruction="Select the most relevant SQL columns to the given text.",
        input_format="Text: {utterance}\n\nColumns:{schema}",
        output_format="{linked_schema}",
        postprocessors=["processors.to_list_by_comma_space"],
    ),
    "templates.schema_linking.default",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        instruction="Select the most relevant SQL columns to the given text. You are also given a hint.",
        input_format="Text: {utterance}\n\nHint: {hint}\n\nColumns:{schema}",
        output_format="{linked_schema}",
        postprocessors=["processors.to_list_by_comma_space"],
    ),
    "templates.schema_linking.with_hint",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        ["templates.schema_linking.default", "templates.schema_linking.with_hint"]
    ),
    "templates.schema_linking.all",
    overwrite=True,
)
