from typing import Any, List

from .operators import FieldOperator


class StringMethods(FieldOperator):
    methods: List[str]
    fields: List[str]

    _possible_methods = {
        "capitalize": [],
        "casefold": [],
        "center": ["width", "fillchar"],
        "count": ["sub", "start", "end"],
        "encode": ["encoding", "errors"],
        "endswith": ["suffix", "start", "end"],
        "expandtabs": ["tabsize"],
        "find": ["sub", "start", "end"],
        "format": ["*args", "**kwargs"],
        "format_map": ["mapping"],
        "index": ["sub", "start", "end"],
        "isalnum": [],
        "isalpha": [],
        "isascii": [],
        "isdecimal": [],
        "isdigit": [],
        "isidentifier": [],
        "islower": [],
        "isnumeric": [],
        "isprintable": [],
        "isspace": [],
        "istitle": [],
        "isupper": [],
        "join": ["iterable"],
        "ljust": ["width", "fillchar"],
        "lower": [],
        "lstrip": ["chars"],
        "maketrans": ["x", "y", "z"],
        "partition": ["sep"],
        "replace": ["old", "new", "count"],
        "rfind": ["sub", "start", "end"],
        "rindex": ["sub", "start", "end"],
        "rjust": ["width", "fillchar"],
        "rpartition": ["sep"],
        "rsplit": ["sep", "maxsplit"],
        "rstrip": ["chars"],
        "split": ["sep", "maxsplit"],
        "splitlines": ["keepends"],
        "startswith": ["prefix", "start", "end"],
        "strip": ["chars"],
        "swapcase": [],
        "title": [],
        "translate": ["table"],
        "upper": [],
        "zfill": ["width"],
    }


class Substring(FieldOperator):
    begin: int = 0
    end: int = None

    def substring(self, s, begin, end):
        if end is None:
            return s[begin:]
        return s[begin:end]

    def process_value(self, text: str) -> Any:
        begin, end = self.get_range(text)
        return self.substring(text, begin, end)
