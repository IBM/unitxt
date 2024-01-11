from src.unitxt import add_to_catalog
from src.unitxt.blocks import ToString, ToStringStripped

operator = ToString(field="TBD")

add_to_catalog(operator, "processors.to_string", overwrite=True)

operator = ToStringStripped(field="TBD")

add_to_catalog(operator, "processors.to_string_stripped", overwrite=True)
