import sys

from unitxt import add_to_catalog
from unitxt.blocks import TemplatesList
from unitxt.templates import InputOutputTemplate

sys.setrecursionlimit(99999)


template_details = [
    (
        "templates.text2sql.you_are_given",
        "You are given the following question:\n\n{utterance}\n\nAn SQL schema\n\n```sql\n\n{schema}\n```\n\nAnswer the following question:\n\n{utterance}\n\n",
        "You are a Text2SQL generation model, in your answer, only have SQL code.\nStart your query with 'SELECT' and end it with ';'\n\n",
    ),
    (
        "templates.text2sql.you_are_given_with_hint",
        "You are given the following question:\n\n{utterance}\n\nAn SQL schema\n\n```sql\n\n{schema}\n```\n\nAnd hint:\n\n{evidence}\n\nAnswer the following question:\n\n{utterance}\n\n",
        "You are a Text2SQL generation model, in your answer, only have SQL code.\nMake sure you start your query with 'SELECT' and end it with ';'\n\n",
    ),
]

template_names = []
for name, input_format, instruction in template_details:
    add_to_catalog(
        InputOutputTemplate(
            input_format=input_format,
            instruction=instruction,
            target_prefix="",
            output_format="{query}",
            postprocessors=["processors.text2sql.get_sql"],
        ),
        name,
        overwrite=True,
    )
    template_names.append(template_names)


add_to_catalog(
    TemplatesList(template_names),
    "templates.text2sql.all",
    overwrite=True,
)
