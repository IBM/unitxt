from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import ICLFormat

format = ICLFormat()

add_to_catalog(format, "formats.empty", overwrite=True)
