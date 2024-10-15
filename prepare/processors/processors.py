import numpy as np
from unitxt import add_to_catalog
from unitxt.logging_utils import get_logger
from unitxt.operator import SequentialOperator
from unitxt.operators import Cast, RemoveValues
from unitxt.processors import (
    Capitalize,
    ConvertToBoolean,
    ExtractArenaHardNumericalJudgment,
    ExtractMtBenchLabelJudgment,
    ExtractMtBenchRatingJudgment,
    ExtractWithRegex,
    FirstCharacter,
    FixWhiteSpace,
    GetStringAfter,
    InferDictsToBinaryLogprobs,
    LiteralEval,
    Lower,
    LowerCaseTillPunc,
    MatchClosestOption,
    PostProcess,
    RegexParser,
    RemoveArticles,
    RemovePunctuations,
    StanceToProCon,
    StringEquals,
    StrToFloatFormat,
    Substring,
    TakeFirstNonEmptyLine,
    TakeFirstWord,
    ToYesOrNone,
    YesNoToInt,
    YesToOneElseZero,
)
from unitxt.settings_utils import get_constants

constants = get_constants()
logger = get_logger()

add_to_catalog(
    PostProcess(TakeFirstNonEmptyLine()),
    "processors.take_first_non_empty_line",
    overwrite=True,
)

add_to_catalog(
    PostProcess(LowerCaseTillPunc()),
    "processors.lower_case_till_punc",
    overwrite=True,
)

add_to_catalog(
    PostProcess(StringEquals(string="hate speech")),
    "processors.hate_speech_or_not_hate_speech",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Lower()),
    "processors.lower_case",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Capitalize()),
    "processors.capitalize",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Substring()),
    "processors.substring",
    overwrite=True,
)

add_to_catalog(
    PostProcess(GetStringAfter(substring=":")),
    "processors.get_string_after_colon",
    overwrite=True,
)

add_to_catalog(
    PostProcess(StringEquals(string="toxic")),
    "processors.toxic_or_not_toxic",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ConvertToBoolean()),
    "processors.convert_to_boolean",
    overwrite=True,
)

add_to_catalog(
    PostProcess(TakeFirstWord()),
    "processors.take_first_word",
    overwrite=True,
)

add_to_catalog(
    PostProcess(YesNoToInt()),
    "processors.yes_no_to_int",
    overwrite=True,
)

add_to_catalog(
    PostProcess(StrToFloatFormat()),
    "processors.str_to_float_format",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ToYesOrNone()),
    "processors.to_yes_or_none",
    overwrite=True,
)

add_to_catalog(
    PostProcess(YesToOneElseZero(), process_references=False),
    "processors.predictions_yes_1_else_0",
    overwrite=True,
)

add_to_catalog(
    PostProcess(StanceToProCon()),
    "processors.stance_to_pro_con",
    overwrite=True,
)

add_to_catalog(
    PostProcess(FirstCharacter()),
    "processors.first_character",
    overwrite=True,
)

add_to_catalog(
    PostProcess(RemoveValues(unallowed_values=["none"])),
    "processors.remove_none_from_list",
    overwrite=True,
)

add_to_catalog(
    PostProcess(MatchClosestOption()),
    "processors.match_closest_option",
    overwrite=True,
)

double_brackets_regex = r"\[\[(.*?)\]\]"
parser = ExtractWithRegex(regex=double_brackets_regex, field="TBD")
example = "A. and also B. And that is why my final answer is [[Yes]]"
logger.info(parser.process_value(example))
assert parser.process_value(example) == "Yes"

add_to_catalog(
    PostProcess(
        ExtractWithRegex(regex=double_brackets_regex), process_references=False
    ),
    "processors.extract_from_double_brackets",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Cast(to="float", failure_default=0.0)),
    "processors.cast_to_float_return_zero_if_failed",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Cast(to="float", failure_default=np.nan)),
    "processors.cast_to_float_return_nan_if_failed",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ExtractMtBenchRatingJudgment()),
    "processors.extract_mt_bench_rating_judgment",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ExtractMtBenchLabelJudgment()),
    "processors.extract_mt_bench_label_judgment",
    overwrite=True,
)

add_to_catalog(
    PostProcess(RegexParser(regex=".+"), process_references=False),
    "processors.regex_parser_from_prediction",
    overwrite=True,
)

add_to_catalog(
    PostProcess(LiteralEval(), process_references=False),
    "processors.literal_eval",
    overwrite=True,
)


add_to_catalog(
    PostProcess(ExtractMtBenchRatingJudgment()),
    "processors.extract_mt_bench_rating_judgment",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ExtractMtBenchLabelJudgment()),
    "processors.extract_mt_bench_label_judgment",
    overwrite=True,
)

add_to_catalog(
    PostProcess(RegexParser(regex=".+"), process_references=False),
    "processors.regex_parser_from_prediction",
    overwrite=True,
)

add_to_catalog(
    PostProcess(LiteralEval(), process_references=False),
    "processors.literal_eval",
    overwrite=True,
)

add_to_catalog(
    PostProcess(Cast(to="float", failure_default={"float": 0.5})),
    "processors.cast_to_float_return_0_5_if_failed",
    overwrite=True,
)

add_to_catalog(
    PostProcess(ExtractArenaHardNumericalJudgment(), process_references=False),
    "processors.extract_arena_hard_numerical_judgment",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            InferDictsToBinaryLogprobs(
                neg_class_name="No",
                pos_class_name="Yes",
                num_logprobs_to_take=3,
                field="prediction",
                process_every_value=False,
            ),
        ]
    ),
    "processors.infer_logprobs_to_yes_no_probs",
    overwrite=True,
)

add_to_catalog(
    SequentialOperator(
        steps=[
            InferDictsToBinaryLogprobs(
                neg_class_name="No",
                pos_class_name="Yes",
                take_logprobs_from_end=True,
                num_logprobs_to_take=3,
                field="prediction",
                process_every_value=False,
            ),
        ]
    ),
    "processors.infer_last_token_logprobs_to_yes_no_probs",
    overwrite=True,
)

add_to_catalog(
    PostProcess(RemoveArticles()),
    "processors.remove_articles",
    overwrite=True,
)

add_to_catalog(
    PostProcess(RemovePunctuations()),
    "processors.remove_punctuations",
    overwrite=True,
)

add_to_catalog(
    PostProcess(FixWhiteSpace()),
    "processors.fix_whitespace",
    overwrite=True,
)
