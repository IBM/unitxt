import json
import re
from typing import Any

from .operator import BaseFieldOperator


class ToString(BaseFieldOperator):
    def process(self, instance):
        return str(instance)


class RegexParser(BaseFieldOperator):
    """
    A processor that uses regex in order to parse a string.
    """

    regex: str
    termination_regex: str = None

    def process(self, text):
        if self.termination_regex is not None and re.fullmatch(self.termination_regex, text):
            return []
        matches = re.findall(self.regex, text)
        return matches


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
                    pair = (key, value) if self.position_key_before_value else (value, key)
                    result.append(pair)
            return result
        except:
            return []


# add_to_catalog(ToString('prediction'), 'processors', 'to_string')
