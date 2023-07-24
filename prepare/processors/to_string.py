from src.unitxt.blocks import (
    ToString,
)
from src.unitxt import add_to_catalog

operator = ToString()

add_to_catalog(operator, 'processors.to_string', overwrite=True)