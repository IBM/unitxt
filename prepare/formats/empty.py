from src.unitxt.catalog import add_to_catalog
from src.unitxt.operators import ModelInputFormatter

format = ModelInputFormatter()

add_to_catalog(format, "formats.empty", overwrite=True)
