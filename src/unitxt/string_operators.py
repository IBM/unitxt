import re
from typing import List

from .operators import FieldOperator


class Split(FieldOperator):
    by: str

    def process_value(self, value: str) -> List[str]:
        return value.split(self.by)


class RegexSplit(FieldOperator):
    by: str

    def process_value(self, value: str) -> List[str]:
        return re.split(self.by, value)


class TokensSplit(FieldOperator):
    model: str
    _requirements_list = ["transformers"]

    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def process_value(self, value: str) -> List[str]:
        return self.tokenizer.tokenize(value)


class Join(FieldOperator):
    by: str

    def process_value(self, value: List[str]) -> str:
        return self.by.join(value)


class Strip(FieldOperator):
    def process_value(self, value: str) -> str:
        return value.strip()


class Replace(FieldOperator):
    old: str
    new: str

    def process_value(self, value: str) -> str:
        return value.replace(self.old, self.new)
