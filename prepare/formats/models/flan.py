from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

add_to_catalog(
    SystemFormat(
        demo_format="{source}\n{target_prefix}{target}\n\n",
        model_input_format=("{demos}" "{source}\n" "{target_prefix}"),
    ),
    "formats.models.flan.few_shot",
    overwrite=True,
)
