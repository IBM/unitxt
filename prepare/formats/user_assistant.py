from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat(
    input_prefix="<|user|>\n",
    output_prefix="<|assistant|>\n",
    input_output_separator="\n",
)

add_to_catalog(format, "formats.user_assistant", overwrite=True)
