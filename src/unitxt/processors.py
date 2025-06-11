import ast
import copy
import json
import re
import string
from difflib import get_close_matches
from typing import Any, Dict

import numpy as np

from .deprecation_utils import deprecation
from .error_utils import Documentation, UnitxtError
from .operator import MultiStreamOperator
from .operators import FieldOperator, InstanceFieldOperator
from .settings_utils import get_constants
from .type_utils import isoftype

constants = get_constants()


class PostProcess(MultiStreamOperator):
    operator: InstanceFieldOperator
    process_prediction: bool = True
    process_references: bool = True

    def prepare(self):
        super().prepare()
        if not isoftype(self.operator, InstanceFieldOperator):
            raise UnitxtError(
                f"PostProcess requires operator field to be of type InstanceFieldOperator. Got object of type <{type(self.operator).__name__}>.",
                Documentation.POST_PROCESSORS,
            )
        self.prediction_operator = copy.copy(self.operator)
        self.prediction_operator.field = "prediction"
        self.references_operator = copy.copy(self.operator)
        self.references_operator.field = "references"
        self.references_operator.process_every_value = True
        self.references_operator.dont_apply_to_streams = [constants.inference_stream]

    def process(self, multi_stream):
        if self.process_prediction:
            multi_stream = self.prediction_operator(multi_stream)
        if self.process_references:
            multi_stream = self.references_operator(multi_stream)
        return multi_stream


class ToString(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return str(text)


class ToStringStripped(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return str(text).strip()


class SplitStrip(FieldOperator):
    delimiter: str = " "
    strip_every_element: bool = False

    def process_value(self, text: Any) -> Any:
        return [
            x.strip() if self.strip_every_element else x
            for x in text.split(self.delimiter)
        ]


class ToListByComma(SplitStrip):
    delimiter = ","
    strip_every_element = True


class ToListByCommaSpace(SplitStrip):
    delimiter = ", "
    strip_every_element = True


class RegexParser(FieldOperator):
    """A processor that uses regex in order to parse a string."""

    regex: str
    termination_regex: str = None

    def process_value(self, text: Any) -> Any:
        if self.termination_regex is not None and re.fullmatch(
            self.termination_regex, text
        ):
            return []
        return re.findall(self.regex, text)


class ExtractWithRegex(RegexParser):
    def process_value(self, text: Any) -> Any:
        matches = super().process_value(text)
        if matches:
            return matches[0]
        return ""


class ListToEmptyEntitiesTuples(FieldOperator):
    def process_value(self, lst: Any) -> Any:
        try:
            return [(str(item), "") for item in lst]
        except json.JSONDecodeError:
            return []


class DictOfListsToPairs(FieldOperator):
    position_key_before_value: bool = True

    def process_value(self, obj: Any) -> Any:
        try:
            result = []
            for key, values in obj.items():
                for value in values:
                    assert isinstance(value, str)
                    pair = (
                        (key, value) if self.position_key_before_value else (value, key)
                    )
                    result.append(pair)
            return result
        except:
            return []


class TakeFirstNonEmptyLine(FieldOperator):
    def process_value(self, text: Any) -> Any:
        parts = str(text).strip().split("\n")
        if len(parts) == 0:
            return ""
        return parts[0].strip()


class TakeLastNonEmptyLine(FieldOperator):
    def process_value(self, text: Any) -> Any:
        parts = str(text).strip().split("\n")
        if len(parts) == 0:
            return ""
        return parts[-1].strip()


class ConvertToBoolean(FieldOperator):
    def process_value(self, text: Any) -> Any:
        clean_instance = str(text).strip().lower()
        if any(w in clean_instance for w in ["no", "not", "wrong", "false"]):
            return "FALSE"
        if any(w in clean_instance for w in ["yes", "right", "correct", "true"]):
            return "TRUE"
        return "OTHER"


class LowerCaseTillPunc(FieldOperator):
    def process_value(self, text: Any) -> Any:
        non_empty_line = text.lower()
        match = re.search(r"[.,!?;]", non_empty_line)
        if match:
            # Extract text up to the first punctuation
            non_empty_line = non_empty_line[: match.start()]
        return non_empty_line


class Lower(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return text.lower()


class Upper(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return str(text).upper()


@deprecation("2.0.0", alternative=Lower)
class LowerCase(Lower):
    pass


class Capitalize(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return text.capitalize()


class GetStringAfter(FieldOperator):
    substring: str

    def process_value(self, text: Any) -> Any:
        return text.split(self.substring, 1)[-1].strip()


class MatchClosestOption(InstanceFieldOperator):
    options_field: str = "options"

    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        options = instance["task_data"][self.options_field]
        return get_close_matches(value, options, n=1, cutoff=0.0)[0]


def process_instance_value(self, value, instance):
    options = instance[self.options_field]
    # Get the closest match; n=1 returns the single closest match
    closest_match = get_close_matches(value, options, n=1, cutoff=0)
    return closest_match[0] if closest_match else None


class Substring(FieldOperator):
    begin: int = 0
    end: int = None

    def process_value(self, text: Any) -> Any:
        if self.end is None:
            return text[self.begin :]
        return text[self.begin : self.end]


class FirstCharacter(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\s*(\w)", text)
        if match:
            return match.groups(0)[0]
        return ""


class TakeFirstWord(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"([-]*[0-9]+(\.([0-9]+))*)|([\w]+)", text)
        if match:
            return text[match.start() : match.end()]
        return ""


class YesNoToInt(FieldOperator):
    def process_value(self, text: Any) -> Any:
        if text == "yes":
            return "1"
        if text == "no":
            return "0"
        return text


class YesToOneElseZero(FieldOperator):
    def process_value(self, text: Any) -> Any:
        if text == "yes":
            return "1"
        return "0"


class StrToFloatFormat(FieldOperator):
    def process_value(self, text: Any) -> Any:
        try:
            return str(float(text))
        except Exception:
            return str(text)


class ToYesOrNone(FieldOperator):
    def process_value(self, text: Any) -> Any:
        if text == "yes":
            return "yes"
        return "none"


class StanceToProCon(FieldOperator):
    def process_value(self, text: Any) -> Any:
        if text == "positive":
            return "PRO"
        if text in ["negative", "suggestion"]:
            return "CON"
        return "none"


class StringEquals(FieldOperator):
    string: str

    def process_value(self, text: Any) -> Any:
        if "not " + self.string.lower() in text.lower():
            return "not " + self.string.lower()
        if self.string.lower() in text.lower():
            return self.string.lower()
        return text


@deprecation("2.0.0", alternative=StringEquals)
class StringOrNotString(StringEquals):
    pass


class ExtractMtBenchRatingJudgment(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\[\[([\d]+\.?[\d]*)\]\]", text)
        try:
            return float(match.group(1)) / 10
        except:
            return 0.0


class ExtractHarmRatingJudgement(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\[\[([\d]+\.?[\d]*)\]\]", text)
        try:
            return float(match.group(1)) * 0.25 - 0.25
        except:
            return np.NaN


class ExtractMtBenchLabelJudgment(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\[\[([^\]]+)\]\]", text)
        try:
            return str(match.group(1))
        except:
            return "None"


class LiteralEval(FieldOperator):
    def process_value(self, text: Any) -> Any:
        if text is not None and not isinstance(text, str):
            raise ValueError(
                f"LiteralEval: field '{self.field}' is expected to be of 'str' input type, got: {type(text)}"
            )
        if text is None or text == "":
            return text
        return ast.literal_eval(text.strip())


class ExtractSafeUnsafeJudgment(FieldOperator):
    def process_value(self, text: Any) -> Any:
        first_line = str(text).strip().split("\n")[0].lower()
        if first_line == "safe":
            return 1.0
        return 0.0


class ExtractArenaHardNumericalJudgment(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\[\[([^\]]+)\]\]", text)
        try:
            res = str(match.group(1))
            if res == "A>B":
                return 1
            if res == "A>>B":
                return 3
            if res == "B>A":
                return -1
            if res == "B>>A":
                return -3
            return 0

        except:
            return 0


class InferDictsToBinaryLogprobs(FieldOperator):
    neg_class_name: str
    pos_class_name: str

    take_logprobs_from_end: bool = False
    num_logprobs_to_take: int = 3
    min_probability_mass = 0.0001

    def verify(self):
        super().verify()
        if (
            self.neg_class_name.lower() in self.pos_class_name.lower()
            or self.pos_class_name.lower() in self.neg_class_name.lower()
        ):
            raise ValueError(
                f"""Class names in {self.__class__.__name__} should not overlap, got "{self.pos_class_name}" and "{self.neg_class_name}"""
            )

    def process_value(self, obj: Any) -> Any:
        for i in self.get_token_range(obj):
            try:
                pos_probs, neg_probs = self.get_pos_neg_probs(pred_dict=obj[i])
                if pos_probs or neg_probs:
                    sum_probs = sum(pos_probs) + sum(neg_probs)
                    if sum_probs > self.min_probability_mass:
                        return sum(pos_probs) / sum_probs
            except:
                pass
        return 0

    def get_pos_neg_probs(self, pred_dict):
        token_logprobs = pred_dict["top_tokens"]

        pos_and_neg_probs = []
        for class_name in [self.pos_class_name, self.neg_class_name]:
            # We need to capture different variants of model behavior and tokenizers, for example with opening space,
            # punctuation etc. but avoid longer words that contain the class name.
            # For example, for class "yes" we would capture "YES," and " Yes" but not "yesterday".
            name_regex = re.compile(
                rf"(\W|Ġ|_)*{class_name}(\W|Ġ|_)*", flags=re.IGNORECASE
            )
            class_probs = [
                np.exp(d["logprob"])
                for d in token_logprobs
                if name_regex.fullmatch(d["text"])
            ]
            pos_and_neg_probs.append(class_probs)
        return pos_and_neg_probs

    def get_token_range(self, obj: Any) -> range:
        n_tokens = min([self.num_logprobs_to_take, len(obj)])
        if self.take_logprobs_from_end:
            return range(-1, -(n_tokens + 1), -1)
        return range(n_tokens)


class RemoveArticles(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return re.sub(r"\b(a|an|the)\b", " ", text)


class RemovePunctuations(FieldOperator):
    def process_value(self, text: Any) -> Any:
        puncs_to_exclude = set(string.punctuation)
        return "".join(c for c in text if c not in puncs_to_exclude)


class FixWhiteSpace(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return " ".join(text.split())


class AddPrefix(FieldOperator):
    prefix: str

    def process_value(self, text: str) -> str:
        text = text.strip()
        if text.startswith(self.prefix):
            return text
        return self.prefix + text.strip()


class GetSQL(FieldOperator):
    """Operator to extract the most likely SQL query from text, often generated by language models.

    It prioritizes SQL within markdown code blocks (```sql or ```)
    and defaults to finding the last SELECT statement in the text
    if no code blocks are found. It attempts to remove trailing text
    after the first semicolon in the identified query.
    """

    def process_value(self, text: str) -> str:
        """Extracts the most plausible SQL query from the given text.

        Args:
            text: The input string potentially containing an SQL query
                  and other text (e.g., explanations, markdown).

        Returns:
            The extracted SQL query string, or a message indicating
            no query was found.
        """
        if not isinstance(text, str):
            return "Input must be a string"  # Basic type check

        sql_query_candidate = None  # Renamed to indicate it might need cleanup

        # 1. Try to find ```sql ... ``` code blocks
        sql_blocks = re.findall(
            r"```sql\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE
        )
        if sql_blocks:
            # Use the content of the last ```sql block
            sql_query_candidate = sql_blocks[-1].strip()
        else:
            # 2. If no ```sql blocks, try to find generic ``` ... ``` blocks
            generic_blocks = re.findall(r"```\s*(.*?)\s*```", text, re.DOTALL)
            if generic_blocks:
                # Check if the last block looks like SQL (starts with SELECT, INSERT, etc.)
                last_block_content = generic_blocks[-1].strip()
                # Allow common SQL starting keywords
                sql_keywords = (
                    r"^(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|WITH|DROP|TRUNCATE)\b"
                )
                if re.match(sql_keywords, last_block_content, re.IGNORECASE):
                    sql_query_candidate = last_block_content

        # 3. If no suitable code blocks found, search the entire text for the last relevant SQL keyword
        if sql_query_candidate is None:
            # Find the start index of the *last* common SQL keyword (case-insensitive)
            last_match = None
            # Expand search beyond just SELECT for better fallback
            sql_keywords_search = (
                r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|WITH|DROP|TRUNCATE)\b"
            )
            for match in re.finditer(sql_keywords_search, text, re.IGNORECASE):
                last_match = match

            if last_match:
                # Extract from the last keyword to the end of the string
                sql_query_candidate = text[last_match.start() :].strip()

        # 4. Cleanup: Truncate at first semicolon and strip whitespace
        if sql_query_candidate:
            # Find the first semicolon in the candidate string
            first_semicolon_index = sql_query_candidate.find(";")
            if first_semicolon_index != -1:
                # If found, take everything before it
                sql_query = sql_query_candidate[:first_semicolon_index].strip()
            else:
                # If no semicolon, use the candidate as is (after stripping)
                sql_query = sql_query_candidate.strip()

            # clean the ```sql\n from the start and the \n``` in case it is there
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        else:
            sql_query = None  # Ensure sql_query is None if no candidate was found

        # 5. Return result or 'not found' message
        return (
            sql_query if sql_query is not None else "No query found in generation"
        )  # Check for None explicitly


class ScaleNumberToZeroOneReturnZeroIfFails(FieldOperator):
    max_val = 10
    min_val = 0

    def process_value(self, text: Any) -> Any:
        try:
            text = float(text)
            return (text - self.min_val) / self.max_val
        except Exception:
            return 0


class ExtractVerbalJudgment(FieldOperator):
    classes = ["not", "somewhat", "mostly", "completely"]

    def process_value(self, text: Any) -> Any:
        max_val = len(self.classes) - 1
        for i, c in enumerate(self.classes):
            if text.strip().lower().startswith(c):
                return i / (max_val)
        return 0


class ExtractVerbalJudgementBadGood(ExtractVerbalJudgment):
    classes = ["very bad", "bad", "mediocre", "good", "very good"]
