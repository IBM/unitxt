import json
from dataclasses import field
from typing import Any, Dict, List, Optional

from datasets import Features, Sequence, Value

from .artifact import Artifact
from .operator import StreamInstanceOperatorValidator

UNITXT_DATASET_SCHEMA = Features(
    {
        "source": Value("string"),
        "target": Value("string"),
        "references": Sequence(Value("string")),
        "metrics": Sequence(Value("string")),
        "group": Value("string"),
        "postprocessors": Sequence(Value("string")),
        "task_data": Value(dtype="string"),
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


def artifact_to_jsonable(artifact):
    if artifact.__id__ is None:
        return artifact.to_dict()
    return artifact.__id__


class ToUnitxtGroup(StreamInstanceOperatorValidator):
    group: str
    metrics: List[str] = None
    postprocessors: List[str] = field(default_factory=lambda: ["to_string_stripped"])
    remove_unnecessary_fields: bool = True

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        metadata = {}
        for key, val in instance["recipe_metadata"]:
            if isinstance(val, Artifact):
                metadata[key] = artifact_to_jsonable(val)

        task_data = {**instance["inputs"], **instance["outputs"], "metadata": metadata}

        instance["task_data"] = json.dumps(task_data)

        if self.remove_unnecessary_fields:
            keys_to_delete = []

            for key in instance.keys():
                if key not in UNITXT_DATASET_SCHEMA:
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del instance[key]
        if "group" not in instance:
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
