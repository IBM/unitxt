from src.unitxt import add_to_catalog
from src.unitxt.processors import (
    FirstCharacter,
    LowerCaseTillPunc,
    StringOrNotString,
    TakeFirstNonEmptyLine,
)

operator = TakeFirstNonEmptyLine()

add_to_catalog(operator, "processors.take_first_non_empty_line", overwrite=True)

operator2 = LowerCaseTillPunc()

add_to_catalog(operator2, "processors.lower_case_till_punc", overwrite=True)

operator3 = StringOrNotString(string="hate speech")

add_to_catalog(operator3, "processors.hate_speech_or_not_hate_speech", overwrite=True)

parser = FirstCharacter()

example = " A. This is the answer."

print(parser.process(example))
assert parser.process(example) == "A"

example = "   "

print(parser.process(example))
assert parser.process(example) == ""

add_to_catalog(parser, "processors.first_character", overwrite=True)
