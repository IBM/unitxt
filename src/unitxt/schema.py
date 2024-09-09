import json
from typing import Any, Dict, List, Optional

from datasets import Audio, Features, Image, Sequence, Value

from .artifact import Artifact
from .dict_utils import dict_get
from .operator import InstanceOperatorValidator
from .settings_utils import get_constants

constants = get_constants()

UNITXT_DATASET_SCHEMA = Features(
    {
        "source": Value("string"),
        "target": Value("string"),
        "references": Sequence(Value("string")),
        "metrics": Sequence(Value("string")),
        "groups": Sequence(Value("string")),
        "subset": Sequence(Value("string")),
        "media": {
            "images": Sequence(Image()),
            "audios": Sequence(Audio()),
        },
        "postprocessors": Sequence(Value("string")),
        "task_data": Value(dtype="string"),
        "data_classification_policy": Sequence(Value("string")),
    }
)

UNITXT_INFERENCE_SCHEMA = Features(
    {
        "source": Value("string"),
        "metrics": Sequence(Value("string")),
        "groups": Sequence(Value("string")),
        "subset": Sequence(Value("string")),
        "postprocessors": Sequence(Value("string")),
        "task_data": Value(dtype="string"),
        "data_classification_policy": Sequence(Value("string")),
    }
)


def get_schema(stream_name):
    if stream_name == constants.inference_stream:
        return UNITXT_INFERENCE_SCHEMA
    return UNITXT_DATASET_SCHEMA


class Finalize(InstanceOperatorValidator):
    group_by: List[List[str]]
    remove_unnecessary_fields: bool = True

    @staticmethod
    def artifact_to_jsonable(artifact):
        if artifact.__id__ is None:
            return artifact.to_dict()
        return artifact.__id__

    def _prepare_media(self, instance):
        if "media" not in instance:
            instance["media"] = {}

        if "images" not in instance["media"]:
            instance["media"]["images"] = []

        if "audios" not in instance["media"]:
            instance["media"]["audios"] = []

        return instance

    def _get_instance_task_data(
        self, instance: Dict[str, Any], use_reference_fields=True
    ) -> Dict[str, Any]:
        task_data = {
            **instance["input_fields"],
            "metadata": {
                "data_classification_policy": instance["data_classification_policy"],
            },
        }
        if use_reference_fields:
            task_data = {**task_data, **instance["reference_fields"]}
        return task_data

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        task_data = self._get_instance_task_data(
            instance,
            use_reference_fields=stream_name != constants.inference_stream,
        )

        task_data["metadata"]["num_demos"] = instance["recipe_metadata"]["num_demos"]
        task_data["metadata"]["template"] = self.artifact_to_jsonable(
            instance["recipe_metadata"]["template"]
        )
        if "demos" in instance:
            task_data["demos"] = [
                self._get_instance_task_data(instance)
                for instance in instance.pop("demos")
            ]

        instance["task_data"] = json.dumps(task_data)

        if self.remove_unnecessary_fields:
            keys_to_delete = []

            for key in instance.keys():
                if key not in get_schema(stream_name):
                    keys_to_delete.append(key)

            for key in keys_to_delete:
                del instance[key]

        data = {**task_data, **task_data["metadata"]}
        groups = []
        for group_attributes in self.group_by:
            group = {}
            if isinstance(group_attributes, str):
                group_attributes = [group_attributes]
            for attribute in group_attributes:
                group[attribute] = dict_get(data, attribute)
            groups.append(json.dumps(group))

        instance["groups"] = groups
        instance["subset"] = []

        instance = self._prepare_media(instance)

        instance["metrics"] = [
            metric.to_json() if isinstance(metric, Artifact) else metric
            for metric in instance["metrics"]
        ]
        instance["postprocessors"] = [
            processor.to_json() if isinstance(processor, Artifact) else processor
            for processor in instance["postprocessors"]
        ]

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
