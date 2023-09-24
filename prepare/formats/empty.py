from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(input_output_separator="\n")

add_to_catalog(format, f"formats.empty", overwrite=True)
