import os
import re
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .operators import FieldOperator, InstanceOperator
from .settings_utils import get_settings
from .utils import retry_connection_with_exponential_backoff

settings = get_settings()


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

        path = self.model
        if settings.hf_offline_models_path is not None:
            path = os.path.join(settings.hf_offline_models_path, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def process_value(self, value: str) -> List[str]:
        return self.tokenizer.tokenize(value)


class TokensSlice(FieldOperator):
    model: str
    start: Optional[int] = None
    stop: Optional[int] = None
    step: Optional[int] = None

    _requirements_list = ["transformers"]

    @retry_connection_with_exponential_backoff(backoff_factor=2)
    def prepare(self):
        super().prepare()
        from transformers import AutoTokenizer

        path = self.model
        if settings.hf_offline_models_path is not None:
            path = os.path.join(settings.hf_offline_models_path, path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)

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


class MapReplace(FieldOperator):
    mapping: Dict[str, str]

    def process_value(self, value: Any) -> Any:
        for key, val in self.mapping.items():
            value = value.replace(key, val)
        return value


class RegexReplace(FieldOperator):
    pattern: str  # A regex pattern
    replacement: str  # The replacement string or template

    def prepare(self):
        super().prepare()
        self.pattern = re.compile(self.pattern)

    def process_value(self, value: Any) -> Any:
        if isinstance(value, str):
            return re.sub(self.pattern, self.replacement, value)
        return value  # If not a string, return the value as is
