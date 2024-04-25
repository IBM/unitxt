from unitxt import add_to_catalog
from unitxt.logging_utils import get_logger
from unitxt.operator import SequentialOperator
from unitxt.operators import RemoveValues
from unitxt.processors import (
    Capitalize,
    ConvertToBoolean,
    ExtractMtBenchJudgment,
    ExtractWithRegex,
    FirstCharacter,
    GetStringAfter,
    LowerCase,
    LowerCaseTillPunc,
    MatchClosestOption,
    StanceToProCon,
    StringOrNotString,
    StrToFloatFormat,
    Substring,
    TakeFirstNonEmptyLine,
    TakeFirstWord,
    ToYesOrNone,
    YesNoToInt,
    YesToOneElseZero,
)

logger = get_logger()

add_to_catalog(
    SequentialOperator(
        steps=[
            TakeFirstNonEmptyLine(field="prediction", process_every_value=False),
            TakeFirstNonEmptyLine(field="references", process_every_value=True),
        ]
    ),
    "processors.take_first_non_empty_line",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            LowerCaseTillPunc(field="prediction", process_every_value=False),
            LowerCaseTillPunc(field="references", process_every_value=True),
        ]
    ),
    "processors.lower_case_till_punc",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            StringOrNotString(
                string="hate speech", field="prediction", process_every_value=False
            ),
            StringOrNotString(
                string="hate speech", field="references", process_every_value=True
            ),
        ]
    ),
    "processors.hate_speech_or_not_hate_speech",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            LowerCase(field="prediction", process_every_value=False),
            LowerCase(field="references", process_every_value=True),
        ]
    ),
    "processors.lower_case",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            Capitalize(field="prediction", process_every_value=False),
            Capitalize(field="references", process_every_value=True),
        ]
    ),
    "processors.capitalize",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            Substring(field="prediction", process_every_value=False),
            Substring(field="references", process_every_value=True),
        ]
    ),
    "processors.substring",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            GetStringAfter(
                substring=":", field="prediction", process_every_value=False
            ),
            GetStringAfter(substring=":", field="references", process_every_value=True),
        ]
    ),
    "processors.get_string_after_colon",
    overwrite=True,
)


add_to_catalog(
    SequentialOperator(
        steps=[
            StringOrNotString(
                string="toxic", field="prediction", process_every_value=False
            ),
            StringOrNotString(
                string="toxic", field="references", process_every_value=True
            ),
        ]
    ),
    "processors.toxic_or_not_toxic",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ConvertToBoolean(field="prediction", process_every_value=False),
            ConvertToBoolean(field="references", process_every_value=True),
        ]
    ),
    "processors.convert_to_boolean",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            TakeFirstWord(field="prediction", process_every_value=False),
            TakeFirstWord(field="references", process_every_value=True),
        ]
    ),
    "processors.take_first_word",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            YesNoToInt(field="prediction", process_every_value=False),
            YesNoToInt(field="references", process_every_value=True),
        ]
    ),
    "processors.yes_no_to_int",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            StrToFloatFormat(field="prediction", process_every_value=False),
            StrToFloatFormat(field="references", process_every_value=True),
        ]
    ),
    "processors.str_to_float_format",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ToYesOrNone(field="prediction", process_every_value=False),
            ToYesOrNone(field="references", process_every_value=True),
        ]
    ),
    "processors.to_yes_or_none",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            YesToOneElseZero(field="prediction", process_every_value=False),
        ]
    ),
    "processors.predictions_yes_1_else_0",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            StanceToProCon(field="prediction", process_every_value=False),
            StanceToProCon(field="references", process_every_value=True),
        ]
    ),
    "processors.stance_to_pro_con",
    overwrite=True,
)


parser = FirstCharacter(field="TBD")
example = " A. This is the answer."
logger.info(parser.process_value(example))
assert parser.process_value(example) == "A"

example = "   "
logger.info(parser.process_value(example))
assert parser.process_value(example) == ""

add_to_catalog(
    SequentialOperator(
        steps=[
            FirstCharacter(field="prediction", process_every_value=False),
            FirstCharacter(field="references", process_every_value=True),
        ]
    ),
    "processors.first_character",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            RemoveValues(
                field="prediction",
                unallowed_values=["none"],
                process_every_value=False,
            ),
            RemoveValues(
                field="references/0",
                unallowed_values=["none"],
                process_every_value=False,
            ),
        ]
    ),
    "processors.remove_none_from_list",
    overwrite=True,
)


add_to_catalog(
    SequentialOperator(
        steps=[
            MatchClosestOption(
                field="prediction",
            ),
            MatchClosestOption(
                field="references",
                process_every_value=True,
            ),
        ]
    ),
    "processors.match_closest_option",
    overwrite=True,
)


double_brackets_regex = r"\[\[(.*?)\]\]"
parser = ExtractWithRegex(regex=double_brackets_regex, field="TBD")
example = "A. and also B. And that is why my final answer is [[Yes]]"
logger.info(parser.process_value(example))
assert parser.process_value(example) == "Yes"


add_to_catalog(
    SequentialOperator(
        steps=[
            ExtractWithRegex(
                regex=double_brackets_regex,
                field="prediction",
                process_every_value=False,
            ),
        ]
    ),
    "processors.extract_from_double_brackets",
)

add_to_catalog(
    SequentialOperator(
        steps=[
            ExtractMtBenchJudgment(
                field="prediction",
            ),
            ExtractMtBenchJudgment(
                field="references",
                process_every_value=True,
            ),
        ]
    ),
    "processors.extract_mt_bench_judgment",
    overwrite=True,
)
