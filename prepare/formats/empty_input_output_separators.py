from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(input_output_separator="")

add_to_catalog(format, f"formats.empty_input_output_separator", overwrite=True)
