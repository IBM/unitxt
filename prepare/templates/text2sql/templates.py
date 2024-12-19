from unitxt import add_to_catalog
from unitxt.blocks import TemplatesList
from unitxt.templates import InputOutputTemplate

add_to_catalog(
    InputOutputTemplate(
        input_format="""You are given the following question:

{utterance}

An SQL schema

```sql
{schema}
```

Answer the following question:

{utterance}

""",
        instruction="You are a Text2SQL generation model, in your answer, only have SQL code.\n"
        "Start your query with 'SELECT' and end it with ';'\n\n",
        target_prefix="",
        output_format="{query}",
        postprocessors=["processors.text2sql.get_sql"],
        # serializer="serializers.text2sql.schema",
    ),
    "templates.text2sql.you_are_given",
    overwrite=True,
)

add_to_catalog(
    InputOutputTemplate(
        input_format="""You are given the following question:

{utterance}

An SQL schema

```sql
{schema}
```

And hint:

{evidence}

Answer the following question:

{utterance}

""",
        instruction="You are a Text2SQL generation model, in your answer, only have SQL code.\n"
        "Make sure you start your query with 'SELECT' and end it with ';'\n\n",
        target_prefix="",
        output_format="{query}",
        postprocessors=["processors.text2sql.get_sql"],
    ),
    "templates.text2sql.you_are_given_with_hint",
    overwrite=True,
)


add_to_catalog(
    TemplatesList(
        [
            "templates.text2sql.you_are_given_with_hint",
            "templates.text2sql.you_are_given",
        ]
    ),
    "templates.text2sql.all",
    overwrite=True,
)
