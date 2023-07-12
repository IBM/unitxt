from dataclasses import field
from typing import Any, Dict, List

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
    postprocessors: List[str] = field(default_factory=lambda: ["to_string"])
    remove_unnecessary_fields: bool = True

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        if self.remove_unnecessary_fields:
            for key in instance.keys():
                if key not in UNITXT_DATASET_SCHEMA:
                    del instance[key]

        instance["group"] = self.group
        if self.metrics is not None:
            instance["metrics"] = self.metrics
        if self.postprocessors is not None:
            instance["postprocessors"] = self.postprocessors

        return instance

    def validate(self, instance: Dict[str, Any], stream_name: str = None):
        # verify the instance has the required schema
        assert instance is not None, f"Instance is None"
        assert isinstance(instance, dict), f"Instance should be a dict, got {type(instance)}"
        assert all(
            [key in instance for key in UNITXT_DATASET_SCHEMA]
        ), f"Instance should have the following keys: {UNITXT_DATASET_SCHEMA}"
        UNITXT_DATASET_SCHEMA.encode_example(instance)
