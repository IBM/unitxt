from typing import List

from .operators import FieldOperator


class Split(FieldOperator):
    by: str

    def process_value(self, value: str) -> List[str]:
        return value.split(self.by)


class Join(FieldOperator):
    by: str

    def process_value(self, value: List[str]) -> str:
        return self.by.join(value)
