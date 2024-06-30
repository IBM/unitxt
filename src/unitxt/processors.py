import ast
import json
import re
from difflib import get_close_matches
from typing import Any, Dict

from .operators import FieldOperator, InstanceFieldOperator


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


class LowerCase(FieldOperator):
    def process_value(self, text: Any) -> Any:
        return text.lower()


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


class StringOrNotString(FieldOperator):
    string: str

    def process_value(self, text: Any) -> Any:
        if "not " + self.string.lower() in text.lower():
            return "not " + self.string.lower()
        if self.string.lower() in text.lower():
            return self.string.lower()
        return text


class ExtractMtBenchRatingJudgment(FieldOperator):
    def process_value(self, text: Any) -> Any:
        match = re.search(r"\[\[([\d]+\.?[\d]*)\]\]", text)
        try:
            return float(match.group(1)) / 10
        except:
            return 0.0


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
