import json
import re

from .operator import BaseFieldOperator


class ToString(BaseFieldOperator):
    def process(self, instance):
        return str(instance)


class ToStringStripped(BaseFieldOperator):
    def process(self, instance):
        return str(instance).strip()


class ToListByComma(BaseFieldOperator):
    def process(self, instance):
        return [x.strip() for x in instance.split(",")]


class RegexParser(BaseFieldOperator):
    """A processor that uses regex in order to parse a string."""

    regex: str
    termination_regex: str = None

    def process(self, text):
        if self.termination_regex is not None and re.fullmatch(
            self.termination_regex, text
        ):
            return []
        return re.findall(self.regex, text)


class LoadJson(BaseFieldOperator):
    def process(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return []


class ListToEmptyEntitiesTuples(BaseFieldOperator):
    def process(self, lst):
        try:
            return [(str(item), "") for item in lst]
        except json.JSONDecodeError:
            return []


class DictOfListsToPairs(BaseFieldOperator):
    position_key_before_value: bool = True

    def process(self, obj):
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


class TakeFirstNonEmptyLine(BaseFieldOperator):
    def process(self, instance):
        splitted = str(instance).strip().split("\n")
        if len(splitted) == 0:
            return ""
        return splitted[0].strip()


class LowerCaseTillPunc(BaseFieldOperator):
    def process(self, instance):
        non_empty_line = instance.lower()
        match = re.search(r"[.,!?;]", non_empty_line)
        if match:
            # Extract text up to the first punctuation
            non_empty_line = non_empty_line[: match.start()]
        return non_empty_line


class LowerCase(BaseFieldOperator):
    def process(self, instance):
        return instance.lower()


class FirstCharacter(BaseFieldOperator):
    def process(self, instance):
        match = re.search(r"\s*(\w)", instance)
        if match:
            return match.groups(0)[0]
        return ""


class StringOrNotString(BaseFieldOperator):
    string: str

    def process(self, instance):
        if "not " + self.string.lower() in instance.lower():
            return "not " + self.string.lower()
        if self.string.lower() in instance.lower():
            return self.string.lower()
        return instance
