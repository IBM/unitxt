from unitxt import add_to_catalog
from unitxt.blocks import TemplatesList
from unitxt.text2sql.templates import Text2SQLInputOutputTemplate

add_to_catalog(
    Text2SQLInputOutputTemplate(
        input_format="You are given the following SQL schema\n\n```sql\n{schema_text}```\n\nand question:\n\n{utterance}\n\n",
        instruction="You are a Text2SQL generation model, in your answer, only have SQL code",
        num_samples=0,
        # use_schema_linking=False,
        use_oracle_knowledge=False,
        db_type="sqlite",
        # target_prefix="Answer:\n```sql\n",
        # target_prefix="SELECT ",
        target_prefix="```sql\nSELECT ",
        output_format="{query}",
        postprocessors=["processors.text2sql.get_sql"],
    ),
    "templates.text2sql.you_are_given",
    overwrite=True,
)

add_to_catalog(
    Text2SQLInputOutputTemplate(
        input_format="""You are given the following question:

{utterance}

An SQL schema

```sql
{schema_text}
```

And hint:

{evidence}

Answer the following question:

{utterance}

""",
        instruction="You are a Text2SQL generation model, in your answer, only have SQL code",
        num_samples=0,
        # use_schema_linking=True,
        use_oracle_knowledge=True,
        db_type="sqlite",
        # target_prefix="Answer:\n```sql\n",
        # target_prefix="SELECT ",
        target_prefix="```sql\nSELECT ",
        output_format="{query}",
        postprocessors=["processors.text2sql.get_sql"],
    ),
    "templates.text2sql.you_are_given_with_hint",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.text2sql.you_are_given",
            "templates.text2sql.you_are_given_with_SL_OK",
        ]
    ),
    "templates.text2sql.all",
    overwrite=True,
)
