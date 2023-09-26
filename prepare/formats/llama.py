from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(
    input_prefix="User: ",
    output_prefix="Agent: ",
    target_prefix="",
    prefix="[INST] <<SYS>>\n",
    suffix="[/INST]",
)

add_to_catalog(format, f"formats.llama", overwrite=True)
