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

    def process(self, instance):
        result = re.search(self.regex, instance)


# add_to_catalog(ToString('prediction'), 'processors', 'to_string')
