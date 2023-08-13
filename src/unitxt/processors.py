import re

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

# add_to_catalog(ToString('prediction'), 'processors', 'to_string')
