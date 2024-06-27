import json
from dataclasses import field
from typing import Any, Dict, List, Optional

from datasets import Features, Sequence, Value

from .operator import InstanceOperatorValidator
from .settings_utils import get_constants

constants = get_constants()

UNITXT_DATASET_SCHEMA = Features(
    {
        "source": Value("string"),
        "target": Value("string"),
        "references": Sequence(Value("string")),
        "metrics": Sequence(Value("string")),
        "group": Value("string"),
        "postprocessors": Sequence(Value("string")),
        "task_data": Value(dtype="string"),
        "data_classification_policy": Sequence(Value("string")),
    }
)


UNITXT_INFERENCE_SCHEMA = Features(
    {
        "source": Value("string"),
        "metrics": Sequence(Value("string")),
        "group": Value("string"),
        "postprocessors": Sequence(Value("string")),
        "task_data": Value(dtype="string"),
        "data_classification_policy": Sequence(Value("string")),
    }
)


def get_schema(stream_name):
    if stream_name == constants.inference_stream:
        return UNITXT_INFERENCE_SCHEMA
    return UNITXT_DATASET_SCHEMA


class ToUnitxtGroup(InstanceOperatorValidator):
    group: str
    metrics: List[str] = None
    postprocessors: List[str] = field(default_factory=lambda: ["to_string_stripped"])
    remove_unnecessary_fields: bool = True

    @staticmethod
    def artifact_to_jsonable(artifact):
        if artifact.__id__ is None:
            return artifact.to_dict()
        return artifact.__id__

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        task_data = {
            **instance["inputs"],
            "metadata": {
                "template": self.artifact_to_jsonable(
                    instance["recipe_metadata"]["template"]
                )
            },
        }

        if stream_name != constants.inference_stream:
            task_data = {**task_data, **instance["outputs"]}

        instance["task_data"] = json.dumps(task_data)

        if self.remove_unnecessary_fields:
            keys_to_delete = []

            for key in instance.keys():
                if key not in get_schema(stream_name):
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
        schema = get_schema(stream_name)
        assert all(
            key in instance for key in schema
        ), f"Instance should have the following keys: {schema}. Instance is: {instance}"
        schema.encode_example(instance)
