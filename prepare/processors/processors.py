from src.unitxt import add_to_catalog
from src.unitxt.processors import TakeFirstNonEmptyLine, LowerCaseTillPunc, HateOrNot

operator = TakeFirstNonEmptyLine()

add_to_catalog(operator, "processors.take_first_non_empty_line", overwrite=True)



operator2 = LowerCaseTillPunc()

add_to_catalog(operator2, "processors.lower_case_till_punc", overwrite=True)

operator3 = HateOrNot()

add_to_catalog(operator3, "processors.hate_or_not", overwrite=True)
