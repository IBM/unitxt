from unitxt.catalog import add_to_catalog
from unitxt.formats import SystemFormat

add_to_catalog(
    SystemFormat(
        demo_format="[EX Q]: {source}\n[EX A]: {target_prefix}{target}\n\n",
        model_input_format=(
            "{instruction}\n\n{demos}[EX Q]: {source}\n[EX A]: {target_prefix}"
        ),
    ),
    "formats.models.flan.exq_exa",
    overwrite=True,
)
