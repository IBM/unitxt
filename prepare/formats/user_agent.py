from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(input_prefix="User: ", output_prefix="Agent: ", input_output_separator="\n")

add_to_catalog(format, f"formats.user_agent", overwrite=True)
