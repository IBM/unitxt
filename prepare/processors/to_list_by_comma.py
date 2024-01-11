from src.unitxt import add_to_catalog
from src.unitxt.processors import ToListByComma

operator = ToListByComma(field="TBD")

add_to_catalog(operator, "processors.to_list_by_comma", overwrite=True)
