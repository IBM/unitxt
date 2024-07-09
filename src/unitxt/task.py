from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from .artifact import fetch_artifact
from .logging_utils import get_logger
from .operator import InstanceOperator
from .type_utils import (
    get_args,
    get_origin,
    isoftype,
    parse_type_string,
    verify_required_schema,
)


class Task(InstanceOperator):
    """Task packs the different instance fields into dictionaries by their roles in the task.

    Attributes:
        inputs (Union[Dict[str, str], List[str]]):
            Dictionary with string names of instance input fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        outputs (Union[Dict[str, str], List[str]]):
            Dictionary with string names of instance output fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        metrics (List[str]): List of names of metrics to be used in the task.
        prediction_type (Optional[str]):
            Need to be consistent with all used metrics. Defaults to None, which means that it will
            be set to Any.
        defaults (Optional[Dict[str, Any]]):
            An optional dictionary with default values for chosen input/output keys. Needs to be
            consistent with names and types provided in 'inputs' and/or 'outputs' arguments.
            Will not overwrite values if already provided in a given instance.

    The output instance contains three fields:
        "inputs" whose value is a sub-dictionary of the input instance, consisting of all the fields listed in Arg 'inputs'.
        "outputs" -- for the fields listed in Arg "outputs".
        "metrics" -- to contain the value of Arg 'metrics'
    """

    inputs: Union[Dict[str, str], List[str]]
    outputs: Union[Dict[str, str], List[str]]
    metrics: List[str]
    prediction_type: Optional[str] = None
    augmentable_inputs: List[str] = []
    defaults: Optional[Dict[str, Any]] = None

    def verify(self):
        for io_type in ["inputs", "outputs"]:
            data = self.inputs if io_type == "inputs" else self.outputs
            if not isoftype(data, Dict[str, str]):
                get_logger().warning(
                    f"'{io_type}' field of Task should be a dictionary of field names and their types. "
                    f"For example, {{'text': 'str', 'classes': 'List[str]'}}. Instead only '{data}' was "
                    f"passed. All types will be assumed to be 'Any'. In future version of unitxt this "
                    f"will raise an exception."
                )
                data = {key: "Any" for key in data}
                if io_type == "inputs":
                    self.inputs = data
                else:
                    self.outputs = data

        if not self.prediction_type:
            get_logger().warning(
                "'prediction_type' was not set in Task. It is used to check the output of "
                "template post processors is compatible with the expected input of the metrics. "
                "Setting `prediction_type` to 'Any' (no checking is done). In future version "
                "of unitxt this will raise an exception."
            )
            self.prediction_type = "Any"

        self.check_metrics_type()

        for augmentable_input in self.augmentable_inputs:
            assert (
                augmentable_input in self.inputs
            ), f"augmentable_input {augmentable_input} is not part of {self.inputs}"

        self.verify_defaults()

    @staticmethod
    @lru_cache(maxsize=None)
    def get_metric_prediction_type(metric_id: str):
        metric = fetch_artifact(metric_id)[0]
        return metric.get_prediction_type()

    def check_metrics_type(self) -> None:
        prediction_type = parse_type_string(self.prediction_type)
        for metric_id in self.metrics:
            metric_prediction_type = Task.get_metric_prediction_type(metric_id)

            if (
                prediction_type == metric_prediction_type
                or prediction_type == Any
                or metric_prediction_type == Any
                or (
                    get_origin(metric_prediction_type) is Union
                    and prediction_type in get_args(metric_prediction_type)
                )
            ):
                continue

            raise ValueError(
                f"The task's prediction type ({prediction_type}) and '{metric_id}' "
                f"metric's prediction type ({metric_prediction_type}) are different."
            )

    def verify_defaults(self):
        if self.defaults:
            if not isinstance(self.defaults, dict):
                raise ValueError(
                    f"If specified, the 'defaults' must be a dictionary, "
                    f"however, '{self.defaults}' was provided instead, "
                    f"which is of type '{type(self.defaults)}'."
                )

            for default_name, default_value in self.defaults.items():
                assert isinstance(default_name, str), (
                    f"If specified, all keys of the 'defaults' must be strings, "
                    f"however, the key '{default_name}' is of type '{type(default_name)}'."
                )

                val_type = self.inputs.get(default_name) or self.outputs.get(
                    default_name
                )

                assert val_type, (
                    f"If specified, all keys of the 'defaults' must refer to a chosen "
                    f"key in either 'inputs' or 'outputs'. However, the name '{default_name}' "
                    f"was provided which does not match any of the keys."
                )

                assert isoftype(default_value, parse_type_string(val_type)), (
                    f"The value of '{default_name}' from the 'defaults' must be of "
                    f"type '{val_type}', however, it is of type '{type(default_value)}'."
                )

    def set_default_values(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        if self.defaults:
            instance = {**self.defaults, **instance}
        return instance

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance = self.set_default_values(instance)

        verify_required_schema(self.inputs, instance)
        verify_required_schema(self.outputs, instance)

        inputs = {key: instance[key] for key in self.inputs.keys()}
        outputs = {key: instance[key] for key in self.outputs.keys()}
        data_classification_policy = instance.get("data_classification_policy", [])

        return {
            "inputs": inputs,
            "outputs": outputs,
            "metrics": self.metrics,
            "data_classification_policy": data_classification_policy,
        }


class FormTask(Task):
    pass
