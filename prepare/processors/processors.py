from src.unitxt import add_to_catalog
from src.unitxt.logging import get_logger
from src.unitxt.processors import (
    ConvertToBoolean,
    FirstCharacter,
    LowerCase,
    LowerCaseTillPunc,
    StringOrNotString,
    TakeFirstNonEmptyLine,
    TakeFirstWord,
    YesNoToInt,
)

logger = get_logger()
operator = TakeFirstNonEmptyLine()

add_to_catalog(operator, "processors.take_first_non_empty_line", overwrite=True)

operator2 = LowerCaseTillPunc()

add_to_catalog(operator2, "processors.lower_case_till_punc", overwrite=True)

operator3 = StringOrNotString(string="hate speech")

add_to_catalog(operator3, "processors.hate_speech_or_not_hate_speech", overwrite=True)

operator4 = LowerCase()

add_to_catalog(operator4, "processors.lower_case", overwrite=True)

operator5 = StringOrNotString(string="toxic")

add_to_catalog(operator5, "processors.toxic_or_not_toxic", overwrite=True)

operator6 = ConvertToBoolean()
add_to_catalog(operator6, "processors.convert_to_boolean", overwrite=True)

operator7 = TakeFirstWord()
add_to_catalog(operator7, "processors.take_first_word", overwrite=True)

operator8 = YesNoToInt()
add_to_catalog(operator8, "processors.yes_no_to_int", overwrite=True)

parser = FirstCharacter()

example = " A. This is the answer."

logger.info(parser.process(example))
assert parser.process(example) == "A"

example = "   "

logger.info(parser.process(example))
assert parser.process(example) == ""

add_to_catalog(parser, "processors.first_character", overwrite=True)
