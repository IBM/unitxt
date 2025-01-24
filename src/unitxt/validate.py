from abc import ABC
from dataclasses import field
from typing import Any, Dict, Optional

from datasets import Features, Sequence, Value

from .operator import InstanceOperator


class Validator(ABC):
    pass


class ValidateSchema(Validator, InstanceOperator):
    schema: Features = None

    def verify(self):
        assert isinstance(
            self.schema, Features
        ), "Schema must be an instance of Features"
        assert self.schema is not None, "Schema must be specified"

    def verify_first_instance(self, instance):
        for std_field in self.standard_fields:
            assert (
                std_field in instance
            ), f'Field "{std_field}" is missing in the first instance'

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return instance


class StandardSchema(Features):
    def __init__(self):
        super().__init__(
            {
                "source": Value("string"),
                "target": Value("string"),
                "references": Sequence(Value("string")),
                "metrics": Sequence(Value("string")),
                "parser": Value("string"),
                # 'group': Value('string'),
                # 'guidance': Value('string'),
            }
        )


class ValidateStandardSchema:
    schema: Features = field(default_factory=StandardSchema)
