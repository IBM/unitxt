from unitxt.catalog import add_to_catalog
from unitxt.templates import MultipleChoiceTemplate

add_to_catalog(
    MultipleChoiceTemplate(
        input_format="{context} Look at the scientific diagram carefully and answer the following question: {question}\n{choices}\nRespond only with the correct option digit.",
        choices_separator="\n",
        target_field="answer",
        enumerator="capitals",
    ),
    "templates.qa.llama_vision.multiple_choice.with_context.ai2d",
    overwrite=True,
)
