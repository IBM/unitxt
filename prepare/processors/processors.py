from src.unitxt import add_to_catalog
from src.unitxt.processors import TakeFirstNonEmptyLine

operator = TakeFirstNonEmptyLine()

add_to_catalog(operator, "processors.take_first_non_empty_line", overwrite=True)
