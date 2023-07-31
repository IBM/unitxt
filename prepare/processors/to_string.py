from src.unitxt import add_to_catalog
from src.unitxt.blocks import ToString

operator = ToString()

add_to_catalog(operator, "processors.to_string", overwrite=True)
