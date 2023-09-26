from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(
    target_prefix="",
    prefix="[INST] ",
    suffix="[/INST]",
)

add_to_catalog(format, f"formats.llama", overwrite=True)
