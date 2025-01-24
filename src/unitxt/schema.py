import json
from typing import Any, Dict, List, Optional

from datasets import Audio, Features, Sequence, Value
from datasets import Image as DatasetImage

from .artifact import Artifact
from .dict_utils import dict_get
from .image_operators import ImageDataString
from .operator import InstanceOperatorValidator
from .settings_utils import get_constants, get_settings
from .type_utils import isoftype
from .types import Image

constants = get_constants()
settings = get_settings()

UNITXT_DATASET_SCHEMA = Features(
    {
        "source": Value("string"),
        "target": Value("string"),
        "references": Sequence(Value("string")),
        "metrics": Sequence(Value("string")),
        "groups": Sequence(Value("string")),
        "subset": Sequence(Value("string")),
        "media": {
            "images": Sequence(DatasetImage()),
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
        "media": {
            "images": Sequence(Image()),
            "audios": Sequence(Audio()),
        },
    }
)


def get_schema(stream_name):
    if stream_name == constants.inference_stream:
        return UNITXT_INFERENCE_SCHEMA
    return UNITXT_DATASET_SCHEMA


def load_chat_source(chat_str):
    chat = json.loads(chat_str)
    for turn in chat:
        if isinstance(turn["content"], list):
            for content in turn["content"]:
                if content["type"] == "image_url":
                    content["image_url"]["url"] = ImageDataString(
                        content["image_url"]["url"]
                    )
    return chat


def loads_instance(batch):
    if (
        "source" in batch
        and isinstance(batch["source"][0], str)
        and (
            batch["source"][0].startswith('[{"role":')
            or batch["source"][0].startswith('[{"content":')
        )
    ):
        batch["source"] = [load_chat_source(d) for d in batch["source"]]
    if (
        not settings.task_data_as_text
        and "task_data" in batch
        and isinstance(batch["task_data"][0], str)
    ):
        batch["task_data"] = [json.loads(d) for d in batch["task_data"]]
    return batch


class FinalizeDataset(InstanceOperatorValidator):
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

        for i in range(len(instance["media"]["images"])):
            if isoftype(instance["media"]["images"][i], Image):
                instance["media"]["images"][i] = instance["media"]["images"][i]["image"]

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

    def serialize_instance_fields(self, instance, task_data):
        if settings.task_data_as_text:
            instance["task_data"] = json.dumps(task_data)

        if not isinstance(instance["source"], str):
            instance["source"] = json.dumps(instance["source"])
        return instance

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        task_data = self._get_instance_task_data(
            instance,
            use_reference_fields=stream_name != constants.inference_stream,
        )

        task_data["metadata"]["num_demos"] = instance["recipe_metadata"]["num_demos"]
        task_data["metadata"]["demos_pool_size"] = instance["recipe_metadata"][
            "demos_pool_size"
        ]
        task_data["metadata"]["template"] = self.artifact_to_jsonable(
            instance["recipe_metadata"]["template"]
        )
        if "criteria" in task_data and isinstance(task_data["criteria"], Artifact):
            task_data["criteria"] = self.artifact_to_jsonable(task_data["criteria"])
        if "demos" in instance:
            task_data["demos"] = [
                self._get_instance_task_data(instance)
                for instance in instance.pop("demos")
            ]

        instance = self.serialize_instance_fields(instance, task_data)

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
