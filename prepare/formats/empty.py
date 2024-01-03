from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import SystemFormat

format = SystemFormat()

add_to_catalog(format, "formats.empty", overwrite=True)
