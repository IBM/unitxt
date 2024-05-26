from typing import Any, Dict, List, Optional

from .operator import InstanceOperator


class NormalizeListFields(InstanceOperator):
    fields: List[str]
    key_prefix: str = ""
    empty_value: str = ""
    separator: str = ", "

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field in self.fields:
            assert field in instance, f"Field {field} not found in instance {instance}"
            assert isinstance(
                instance[field], list
            ), f"Field {field} should be a list, got {type(instance[field])}"

            target_key = self.key_prefix + field

            if len(instance[field]) == 0:
                instance[target_key] = self.empty_value
            elif len(instance[field]) == 1:
                instance[target_key] = instance[field][0]
            else:
                instance[target_key] = self.separator.join(instance[field])

        return instance
