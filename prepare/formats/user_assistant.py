from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ModelInputFormatter

format = ModelInputFormatter(
    demo_format="<|user|>\n{source}\n<|assistant|>\n {target}\n\n",
    model_input_format="{system_prompt}{instruction}\n{demos}\n<|user|>\n{source}\n<|assistant|>\n",
)

add_to_catalog(format, "formats.user_assistant", overwrite=True)
