from dataclasses import field
from typing import Any, Dict, List, Optional

from datasets import Features, Sequence, Value

from .operator import StreamInstanceOperatorValidator

UNITXT_DATASET_SCHEMA = Features(
    {
        "source": Value("string"),
        "target": Value("string"),
        "references": Sequence(Value("string")),
        "metrics": Sequence(Value("string")),
        "group": Value("string"),
        "postprocessors": Sequence(Value("string")),
        "additional_inputs": Sequence(
            {"key": Value(dtype="string"), "value": Value("string")}
        ),
    }
)

# UNITXT_METRIC_SCHEMA = Features({
#     "predictions": Value("string", id="sequence"),
#     "target": Value("string", id="sequence"),
#     "references": Value("string", id="sequence"),
#     "metrics": Value("string", id="sequence"),
#     'group': Value('string'),
#     'postprocessors': Value("string", id="sequence"),
# })


class ToUnitxtGroup(StreamInstanceOperatorValidator):
    group: str
    metrics: List[str] = None
    postprocessors: List[str] = field(default_factory=lambda: ["to_string_stripped"])
    remove_unnecessary_fields: bool = True

    def _to_lists_of_keys_and_values(self, dict: Dict[str, str]):
        return {
            "key": [key for key, _ in dict.items()],
            "value": [str(value) for _, value in dict.items()],
        }

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        additional_inputs = {**instance["inputs"], **instance["outputs"]}
        instance["additional_inputs"] = self._to_lists_of_keys_and_values(
            additional_inputs
        )

        if self.remove_unnecessary_fields:
            keys_to_delete = []

            for key in instance.keys():
                if key not in UNITXT_DATASET_SCHEMA:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del instance[key]
        instance["group"] = self.group
        if self.metrics is not None:
            instance["metrics"] = self.metrics
        if self.postprocessors is not None:
            instance["postprocessors"] = self.postprocessors
        return instance

    def validate(self, instance: Dict[str, Any], stream_name: Optional[str] = None):
        # verify the instance has the required schema
        assert instance is not None, "Instance is None"
        assert isinstance(
            instance, dict
        ), f"Instance should be a dict, got {type(instance)}"
        assert all(
            key in instance for key in UNITXT_DATASET_SCHEMA
        ), f"Instance should have the following keys: {UNITXT_DATASET_SCHEMA}. Instance is: {instance}"
        UNITXT_DATASET_SCHEMA.encode_example(instance)
