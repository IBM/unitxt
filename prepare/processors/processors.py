import numpy as np
from unitxt import add_to_catalog
from unitxt.logging_utils import get_logger
from unitxt.operator import SequentialOperator
from unitxt.operators import Cast, InstanceFieldOperator, RemoveValues
from unitxt.processors import (
    Capitalize,
    ConvertToBoolean,
    ExtractArenaHardNumericalJudgment,
    ExtractMtBenchLabelJudgment,
    ExtractMtBenchRatingJudgment,
    ExtractVerbalJudgementBadGood,
    ExtractVerbalJudgment,
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
    ScaleNumberToZeroOneReturnZeroIfFails,
    StanceToProCon,
    StringEquals,
    StrToFloatFormat,
    Substring,
    TakeFirstNonEmptyLine,
    TakeFirstWord,
    TakeLastNonEmptyLine,
    ToYesOrNone,
    Upper,
    YesNoToInt,
    YesToOneElseZero,
)
from unitxt.settings_utils import get_constants

constants = get_constants()
logger = get_logger()


def add_processor_and_operator_to_catalog(
    artifact_name: str,
    operator: InstanceFieldOperator,
    process_references: bool = True,
    process_prediction: bool = True,
    overwrite: bool = True,
):
    """Adds a processor and its associated operator to the catalog.

    Args:
        artifact_name (str): The name of the artifact to associate with the processor and operator.
        operator (InstanceFieldOperator): The operator instance to be added.
        process_references (bool, optional): Whether to process references or not. Defaults to True similar to PostProcess.
        process_prediction (bool, optional): Whether to process the prediction or not. Defaults to True similar to PostProcess.
        overwrite (bool, optional): Whether to overwrite an existing entry with the same artifact name. Defaults to True.
    """
    add_to_catalog(
        PostProcess(
            operator,
            process_references=process_references,
            process_prediction=process_prediction,
        ),
        f"processors.{artifact_name}",
        overwrite=overwrite,
    )
    add_to_catalog(operator, f"operators.{artifact_name}", overwrite=overwrite)


add_processor_and_operator_to_catalog(
    artifact_name="take_first_non_empty_line",
    operator=TakeFirstNonEmptyLine(),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="take_last_non_empty_line",
    operator=TakeLastNonEmptyLine(),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="literal_eval",
    operator=LiteralEval(),
    process_references=False,
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="lower_case_till_punc", operator=LowerCaseTillPunc(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="hate_speech_or_not_hate_speech",
    operator=StringEquals(string="hate speech"),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="lower_case", operator=Lower(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="upper_case", operator=Upper(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="capitalize", operator=Capitalize(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="substring", operator=Substring(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="get_string_after_colon",
    operator=GetStringAfter(substring=":"),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="toxic_or_not_toxic",
    operator=StringEquals(string="toxic"),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="convert_to_boolean", operator=ConvertToBoolean(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="take_first_word", operator=TakeFirstWord(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="yes_no_to_int", operator=YesNoToInt(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="str_to_float_format", operator=StrToFloatFormat(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="to_yes_or_none", operator=ToYesOrNone(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="predictions_yes_1_else_0",
    operator=YesToOneElseZero(),
    process_references=False,
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="stance_to_pro_con", operator=StanceToProCon(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="first_character", operator=FirstCharacter(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="remove_none_from_list",
    operator=RemoveValues(unallowed_values=["none"]),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="match_closest_option", operator=MatchClosestOption(), overwrite=True
)

double_brackets_regex = r"\[\[(.*?)\]\]"
parser = ExtractWithRegex(regex=double_brackets_regex, field="TBD")
example = "A. and also B. And that is why my final answer is [[Yes]]"
logger.info(parser.process_value(example))
assert parser.process_value(example) == "Yes"


add_processor_and_operator_to_catalog(
    artifact_name="extract_from_double_brackets",
    operator=ExtractWithRegex(regex=double_brackets_regex),
    process_references=False,
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="cast_to_float_return_zero_if_failed",
    operator=Cast(to="float", failure_default=0.0),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="cast_to_float_return_nan_if_failed",
    operator=Cast(to="float", failure_default=np.nan),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="extract_mt_bench_rating_judgment",
    operator=ExtractMtBenchRatingJudgment(),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="extract_mt_bench_label_judgment",
    operator=ExtractMtBenchLabelJudgment(),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="cast_to_float_return_0_5_if_failed",
    operator=Cast(to="float", failure_default=0.5),
    overwrite=True,
)

add_processor_and_operator_to_catalog(
    artifact_name="extract_arena_hard_numerical_judgment",
    operator=ExtractArenaHardNumericalJudgment(),
    process_references=False,
    overwrite=True,
)

add_to_catalog(
    PostProcess(RegexParser(regex=".+"), process_references=False),
    "processors.regex_parser_from_prediction",
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

add_processor_and_operator_to_catalog(
    artifact_name="remove_articles", operator=RemoveArticles(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="remove_punctuations", operator=RemovePunctuations(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="fix_whitespace", operator=FixWhiteSpace(), overwrite=True
)

add_processor_and_operator_to_catalog(
    artifact_name="scale_0_10_to_0_1",
    operator=ScaleNumberToZeroOneReturnZeroIfFails(),
    overwrite=True,
    process_references=False,
)

add_processor_and_operator_to_catalog(
    artifact_name="extract_verbal_judgement",
    operator=ExtractVerbalJudgment(),
    overwrite=True,
    process_references=False,
)

add_processor_and_operator_to_catalog(
    artifact_name="extract_verbal_judgement_bad_good",
    operator=ExtractVerbalJudgementBadGood(),
    overwrite=True,
    process_references=False,
)
