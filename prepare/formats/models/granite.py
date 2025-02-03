from unitxt.catalog import add_to_catalog
from unitxt.formats import GraniteDocumentsFormat

format = GraniteDocumentsFormat(model="ibm-granite/granite-3.1-8b-instruct")

add_to_catalog(format, "formats.models.granite_3_1_documents", overwrite=True)
