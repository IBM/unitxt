import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .operators import FieldOperator, InstanceOperator


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


class TokensSlice(FieldOperator):
    model: str
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    _requirements_list = ["transformers"]

    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def process_value(self, value: str) -> str:
        encoded = self.tokenizer.encode(value)
        slicer = slice(self.start, self.stop, self.step)
        sliced = encoded[slicer]
        return self.tokenizer.decode(sliced)


class Join(FieldOperator):
    by: str

    def process_value(self, value: List[str]) -> str:
        return self.by.join(value)


class FormatText(InstanceOperator):
    to_field: str
    text: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance[self.to_field] = self.text.format(**instance)
        return instance


class Strip(FieldOperator):
    def process_value(self, value: str) -> str:
        return value.strip()


class Replace(FieldOperator):
    old: str
    new: str

    def process_value(self, value: str) -> str:
        return value.replace(self.old, self.new)
