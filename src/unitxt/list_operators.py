from typing import Any, Dict, List, Optional

from .operators import (
    FieldOperator,
    ZipFieldValues,
)


class ListsToListOfDicts(ZipFieldValues):
    with_keys: List[str]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance = super().process(instance)
        tuples = instance[self.to_field]
        result = []
        for tuple in tuples:
            result.append(dict(zip(self.with_keys, tuple)))
        instance[self.to_field] = result
        return instance


class WrapWithList(FieldOperator):
    def process_value(self, value: Any) -> Any:
        return [value]


class SelectRange(FieldOperator):
    begin: int = 0
    end: int = None

    def process_value(self, text: Any) -> Any:
        if self.end is None:
            return text[self.begin :]
        return text[self.begin : self.end]
