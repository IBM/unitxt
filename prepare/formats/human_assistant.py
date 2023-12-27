from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ModelInputFormatter

format = ModelInputFormatter(
    demo_format="Human: {source}\nAssistant:  {target}\n\n",
    model_input_format="{system_prompt}{instruction}\n{demos}\nHuman: {source}\nAssistant: ",
)

add_to_catalog(format, "formats.human_assistant", overwrite=True)
