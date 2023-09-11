from src.unitxt.formats import ICLFormat
from src.unitxt.catalog import add_to_catalog

format = ICLFormat(
    input_prefix="User:",
    output_prefix="Agent:",
)

add_to_catalog(format, f"formats.user_agent", overwrite=True)