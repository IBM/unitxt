from unitxt import add_to_catalog
from unitxt.blocks import TemplatesList
from unitxt.text2sql.templates import Text2SQLInputOutputTemplate

add_to_catalog(
    Text2SQLInputOutputTemplate(
        input_format="Question:\nYou are given the following SQL schema\n\n```sql\n{schema_text}```\n\n{utterance}\n\n\n",
        instruction="",
        num_samples=0,
        use_schema_linking=False,
        use_oracle_knowledge=False,
        db_type="sqlite",
        target_prefix="Answer:\n```sql\n",
        output_format="{query}",
        postprocessors=["processors.text2sql.get_sql"],
    ),
    "templates.text2sql.you_are_given",
    overwrite=True,
)

add_to_catalog(
    TemplatesList(
        [
            "templates.text2sql.you_are_given",
        ]
    ),
    "templates.text2sql.all",
    overwrite=True,
)
