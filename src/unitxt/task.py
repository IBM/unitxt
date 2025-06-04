import warnings
from typing import Any, Dict, List, Optional, Union

from .artifact import fetch_artifact
from .deprecation_utils import deprecation
from .error_utils import Documentation, UnitxtError, UnitxtWarning
from .logging_utils import get_logger
from .metrics import MetricsList
from .operator import InstanceOperator
from .operators import ArtifactFetcherMixin
from .settings_utils import get_constants, get_settings
from .templates import Template
from .type_utils import (
    Type,
    get_args,
    get_origin,
    is_type_dict,
    isoftype,
    parse_type_dict,
    parse_type_string,
    to_type_dict,
    to_type_string,
    verify_required_schema,
)

constants = get_constants()
logger = get_logger()
settings = get_settings()


@deprecation(
    version="2.0.0",
    msg="use python type instead of type strings (e.g Dict[str] instead of 'Dict[str]')",
)
def parse_string_types_instead_of_actual_objects(obj):
    if isinstance(obj, dict):
        return parse_type_dict(obj)
    return parse_type_string(obj)


class Task(InstanceOperator, ArtifactFetcherMixin):
    """Task packs the different instance fields into dictionaries by their roles in the task.

    Args:
        input_fields (Union[Dict[str, str], List[str]]):
            Dictionary with string names of instance input fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        reference_fields (Union[Dict[str, str], List[str]]):
            Dictionary with string names of instance output fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        metrics (List[str]):
            List of names of metrics to be used in the task.
        prediction_type (Optional[str]):
            Need to be consistent with all used metrics. Defaults to None, which means that it will
            be set to Any.
        defaults (Optional[Dict[str, Any]]):
            An optional dictionary with default values for chosen input/output keys. Needs to be
            consistent with names and types provided in 'input_fields' and/or 'output_fields' arguments.
            Will not overwrite values if already provided in a given instance.

    The output instance contains three fields:
        1. "input_fields" whose value is a sub-dictionary of the input instance, consisting of all the fields listed in Arg 'input_fields'.
        2. "reference_fields" -- for the fields listed in Arg "reference_fields".
        3. "metrics" -- to contain the value of Arg 'metrics'
    """

    input_fields: Optional[Union[Dict[str, Type], Dict[str, str], List[str]]] = None
    reference_fields: Optional[Union[Dict[str, Type], Dict[str, str], List[str]]] = None
    inputs: Optional[Union[Dict[str, Type], Dict[str, str], List[str]]] = None
    outputs: Optional[Union[Dict[str, Type], Dict[str, str], List[str]]] = None
    metrics: List[str]
    prediction_type: Optional[Union[Type, str]] = None
    augmentable_inputs: List[str] = []
    defaults: Optional[Dict[str, Any]] = None
    default_template: Template = None

    def prepare_args(self):
        super().prepare_args()
        if isinstance(self.metrics, str):
            self.metrics = [self.metrics]

        if self.input_fields is not None and self.inputs is not None:
            raise UnitxtError(
                "Conflicting attributes: 'input_fields' cannot be set simultaneously with 'inputs'. Use only 'input_fields'",
                Documentation.ADDING_TASK,
            )
        if self.reference_fields is not None and self.outputs is not None:
            raise UnitxtError(
                "Conflicting attributes: 'reference_fields' cannot be set simultaneously with 'output'. Use only 'reference_fields'",
                Documentation.ADDING_TASK,
            )

        if self.default_template is not None and not isoftype(
            self.default_template, Template
        ):
            raise UnitxtError(
                f"The task's 'default_template' attribute is not of type Template. The 'default_template' attribute is of type {type(self.default_template)}: {self.default_template}",
                Documentation.ADDING_TASK,
            )

        self.input_fields = (
            self.input_fields if self.input_fields is not None else self.inputs
        )
        self.reference_fields = (
            self.reference_fields if self.reference_fields is not None else self.outputs
        )

        if isoftype(self.input_fields, Dict[str, str]):
            self.input_fields = parse_string_types_instead_of_actual_objects(
                self.input_fields
            )
        if isoftype(self.reference_fields, Dict[str, str]):
            self.reference_fields = parse_string_types_instead_of_actual_objects(
                self.reference_fields
            )

        if isinstance(self.prediction_type, str):
            self.prediction_type = parse_string_types_instead_of_actual_objects(
                self.prediction_type
            )

        if hasattr(self, "inputs") and self.inputs is not None:
            self.inputs = self.input_fields

        if hasattr(self, "outputs") and self.outputs is not None:
            self.outputs = self.reference_fields

    def task_deprecations(self):
        if hasattr(self, "inputs") and self.inputs is not None:
            depr_message = (
                "The 'inputs' field is deprecated. Please use 'input_fields' instead."
            )
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)
        if hasattr(self, "outputs") and self.outputs is not None:
            depr_message = "The 'outputs' field is deprecated. Please use 'reference_fields' instead."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def verify(self):
        self.task_deprecations()

        if self.input_fields is None:
            raise UnitxtError(
                "Missing attribute in task: 'input_fields' not set.",
                Documentation.ADDING_TASK,
            )
        if self.reference_fields is None:
            raise UnitxtError(
                "Missing attribute in task: 'reference_fields' not set.",
                Documentation.ADDING_TASK,
            )
        for io_type in ["input_fields", "reference_fields"]:
            data = (
                self.input_fields
                if io_type == "input_fields"
                else self.reference_fields
            )

            if isinstance(data, list) or not is_type_dict(data):
                UnitxtWarning(
                    f"'{io_type}' field of Task should be a dictionary of field names and their types. "
                    f"For example, {{'text': str, 'classes': List[str]}}. Instead only '{data}' was "
                    f"passed. All types will be assumed to be 'Any'. In future version of unitxt this "
                    f"will raise an exception.",
                    Documentation.ADDING_TASK,
                )
                if isinstance(data, dict):
                    data = parse_type_dict(to_type_dict(data))
                else:
                    data = {key: Any for key in data}

                if io_type == "input_fields":
                    self.input_fields = data
                else:
                    self.reference_fields = data

        if not self.prediction_type:
            UnitxtWarning(
                "'prediction_type' was not set in Task. It is used to check the output of "
                "template post processors is compatible with the expected input of the metrics. "
                "Setting `prediction_type` to 'Any' (no checking is done). In future version "
                "of unitxt this will raise an exception.",
                Documentation.ADDING_TASK,
            )
            self.prediction_type = Any

        self.check_metrics_type()

        for augmentable_input in self.augmentable_inputs:
            assert (
                augmentable_input in self.input_fields
            ), f"augmentable_input {augmentable_input} is not part of {self.input_fields}"

        self.verify_defaults()

    @classmethod
    def process_data_after_load(cls, data):
        possible_dicts = ["inputs", "input_fields", "outputs", "reference_fields"]
        for dict_name in possible_dicts:
            if dict_name in data and isinstance(data[dict_name], dict):
                data[dict_name] = parse_type_dict(data[dict_name])
        if "prediction_type" in data:
            data["prediction_type"] = parse_type_string(data["prediction_type"])
        return data

    def process_data_before_dump(self, data):
        possible_dicts = ["inputs", "input_fields", "outputs", "reference_fields"]
        for dict_name in possible_dicts:
            if dict_name in data and isinstance(data[dict_name], dict):
                if not isoftype(data[dict_name], Dict[str, str]):
                    data[dict_name] = to_type_dict(data[dict_name])
        if "prediction_type" in data:
            if not isinstance(data["prediction_type"], str):
                data["prediction_type"] = to_type_string(data["prediction_type"])
        return data

    @classmethod
    def get_metrics_artifact_without_load(cls, metric_id: str):
        with settings.context(skip_artifacts_prepare_and_verify=True):
            metric, _ = fetch_artifact(metric_id)
        if isinstance(metric, MetricsList):
            return metric.items
        return [metric]

    def check_metrics_type(self) -> None:
        prediction_type = self.prediction_type
        for metric_id in self.metrics:
            metric_artifacts_list = Task.get_metrics_artifact_without_load(metric_id)
            for metric_artifact in metric_artifacts_list:
                metric_prediction_type = metric_artifact.prediction_type
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

                raise UnitxtError(
                    f"The task's prediction type ({prediction_type}) and '{metric_id}' "
                    f"metric's prediction type ({metric_prediction_type}) are different.",
                    Documentation.ADDING_TASK,
                )

    def verify_defaults(self):
        if self.defaults:
            if not isinstance(self.defaults, dict):
                raise UnitxtError(
                    f"If specified, the 'defaults' must be a dictionary, "
                    f"however, '{self.defaults}' was provided instead, "
                    f"which is of type '{to_type_string(type(self.defaults))}'.",
                    Documentation.ADDING_TASK,
                )

            for default_name, default_value in self.defaults.items():
                assert isinstance(default_name, str), (
                    f"If specified, all keys of the 'defaults' must be strings, "
                    f"however, the key '{default_name}' is of type '{to_type_string(type(default_name))}'."
                )

                val_type = self.input_fields.get(
                    default_name
                ) or self.reference_fields.get(default_name)

                assert val_type, (
                    f"If specified, all keys of the 'defaults' must refer to a chosen "
                    f"key in either 'input_fields' or 'reference_fields'. However, the name '{default_name}' "
                    f"was provided which does not match any of the keys."
                )

                assert isoftype(default_value, val_type), (
                    f"The value of '{default_name}' from the 'defaults' must be of "
                    f"type '{to_type_string(val_type)}', however, it is of type '{to_type_string(type(default_value))}'."
                )

    def set_default_values(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        if self.defaults:
            instance = {**self.defaults, **instance}
        return instance

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance = self.set_default_values(instance)

        verify_required_schema(
            self.input_fields,
            instance,
            class_name="Task",
            id=self.__id__,
            description=self.__description__,
        )
        input_fields = {key: instance[key] for key in self.input_fields.keys()}
        data_classification_policy = instance.get("data_classification_policy", [])

        result = {
            "input_fields": input_fields,
            "metrics": self.metrics,
            "data_classification_policy": data_classification_policy,
            "media": instance.get("media", {}),
            "recipe_metadata": instance.get("recipe_metadata", {}),
        }
        if constants.demos_field in instance:
            # for the case of recipe.skip_demoed_instances
            result[constants.demos_field] = instance[constants.demos_field]

        if constants.instruction_field in instance:
            result[constants.instruction_field] = instance[constants.instruction_field]

        if constants.system_prompt_field in instance:
            result[constants.system_prompt_field] = instance[
                constants.system_prompt_field
            ]

        if stream_name == constants.inference_stream:
            return result

        verify_required_schema(
            self.reference_fields,
            instance,
            class_name="Task",
            id=self.__id__,
            description=self.__description__,
        )
        result["reference_fields"] = {
            key: instance[key] for key in self.reference_fields.keys()
        }

        return result


@deprecation(version="2.0.0", alternative=Task)
class FormTask(Task):
    pass
