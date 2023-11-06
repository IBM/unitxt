from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(
    input_prefix="input: ",
    output_prefix="output: ",
)

add_to_catalog(format, f"formats.input_output_prefix", overwrite=True)
