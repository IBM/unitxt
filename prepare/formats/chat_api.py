from unitxt.catalog import add_to_catalog
from unitxt.formats import ChatAPIFormat

format = ChatAPIFormat()

add_to_catalog(format, "formats.chat_api", overwrite=True)
