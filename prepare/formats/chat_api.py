from unitxt.catalog import add_to_catalog
from unitxt.formats import OpenAIFormat

format = OpenAIFormat()

add_to_catalog(format, "formats.chat_api", overwrite=True)
