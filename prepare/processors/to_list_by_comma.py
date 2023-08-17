from src.unitxt import add_to_catalog
from src.unitxt.processors import ToListByComma

operator = ToListByComma()

add_to_catalog(operator, "processors.to_list_by_comma", overwrite=True)
