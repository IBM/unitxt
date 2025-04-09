from unitxt import add_to_catalog
from unitxt.formats import HFSystemFormat

format = HFSystemFormat(model_name="ibm-granite/granite-3.1-2b-instruct")

add_to_catalog(format, "formats.chat_api_with_tokenizer_chat_template", overwrite=True)
