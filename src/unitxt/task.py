from typing import Any, Dict, List, Literal, Optional, Union

from .artifact import fetch_artifact
from .logging_utils import get_logger
from .operator import StreamInstanceOperator
from .type_utils import isoftype, parse_type_string


class Tasker:
    pass


class FormTask(Tasker, StreamInstanceOperator):
    """FormTask packs the different instance fields into dictionaries by their roles in the task.

    Attributes:
        inputs (Union[Dict[str, Any], List[str]]):
            Dictionary with string names of instance input fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        outputs (Union[Dict[str, Any], List[str]]):
            Dictionary with string names of instance output fields and types of respective values.
            In case a list is passed, each type will be assumed to be Any.
        metrics (List[str]): List of names of metrics to be used in the task.
        prediction_type (Optional[Any]):
            Need to be consistent with all used metrics. Defaults to None, which means that it will
            be inferred based on prediction type of specified metrics.

    The output instance contains three fields:
        "inputs" whose value is a sub-dictionary of the input instance, consisting of all the fields listed in Arg 'inputs'.
        "outputs" -- for the fields listed in Arg "outputs".
        "metrics" -- to contain the value of Arg 'metrics'
    """

    inputs: Union[Dict[str, Any], List[str]]
    outputs: Union[Dict[str, Any], List[str]]
    metrics: List[str]
    prediction_type: Optional[Any] = None
    augmentable_inputs: List[str] = []

    @staticmethod
    def process_type(data_type: Any) -> Any:
        if isinstance(data_type, str):
            data_type = data_type.replace("typing.", "")
            return parse_type_string(data_type)
        return data_type

    def verify(self):
        for io_type in ["inputs", "outputs"]:
            data = self.inputs if io_type == "inputs" else self.outputs
            if not isinstance(data, Dict):
                get_logger().warning(
                    f"'{io_type}' should be a dict of field names and their types, instead only "
                    f"'{data}' was passed. All types will be assumed to be 'Any'. In future version "
                    f"of unitxt this will raise an exception."
                )
                data = {key: Any for key in data}
                if io_type == "inputs":
                    self.inputs = data
                else:
                    self.outputs = data

        if not self.prediction_type:
            get_logger().warning(
                "'prediction_type' was not passed. It must be compatible with used metrics. "
                "Trying to infer `prediction_type` based on specified metrics. In future "
                "version of unitxt this will raise an exception."
            )
            sample_metric = fetch_artifact(self.metrics[0])[0]
            self.prediction_type = sample_metric.get_prediction_type()

        for augmentable_input in self.augmentable_inputs:
            assert (
                augmentable_input in self.inputs
            ), f"augmentable_input f{augmentable_input} is not part of {self.inputs}"

    def check_metrics_type(self) -> None:
        for metric_name in self.metrics:
            metric = fetch_artifact(metric_name)[0]
            metric_prediction_type = metric.get_prediction_type()

            if self.prediction_type == metric_prediction_type:
                continue

            prediction_type = self.process_type(self.prediction_type)
            metric_prediction_type = self.process_type(metric_prediction_type)
            if (
                prediction_type == metric_prediction_type
                or prediction_type == Any
                or metric_prediction_type == Any
            ):
                continue

            raise ValueError(
                f"Given prediction type '{prediction_type}' and metric '{metric_name}' "
                f"prediction type '{metric_prediction_type}' are different."
            )

    def access_instance_value(
        self, instance: Dict[str, Any], key: str, io_type: Literal["inputs", "outputs"]
    ) -> Any:
        try:
            return instance[key]
        except KeyError as e:
            io = self.inputs if io_type == "inputs" else self.outputs
            raise KeyError(
                f"Unexpected FormTask {io_type} column names ({[k for k in io if k not in instance]}). "
                f"The available {io_type} names: {list(instance.keys())}"
            ) from e

    def check_instance_value_type(
        self,
        value: Any,
        data_type: Any,
        value_name: str,
        io_type: Literal["inputs", "outputs"],
    ) -> Any:
        data_type = self.process_type(data_type)

        if isoftype(value, data_type):
            return value

        raise ValueError(
            f"Passed {io_type} value {value} under key {value_name} is not of required "
            f"type {data_type}."
        )

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        self.check_metrics_type()

        inputs = {
            key: self.check_instance_value_type(
                self.access_instance_value(instance, key, "inputs"),
                data_type,
                key,
                "inputs",
            )
            for key, data_type in self.inputs.items()
        }
        outputs = {
            key: self.check_instance_value_type(
                self.access_instance_value(instance, key, "outputs"),
                data_type,
                key,
                "outputs",
            )
            for key, data_type in self.outputs.items()
        }

        return {
            "inputs": inputs,
            "outputs": outputs,
            "metrics": self.metrics,
        }


class MultipleChoiceTask(FormTask):
    choices_field: str = "choices"
    choices_separator: str = "\n"
    enumeration_suffix: str = ". "
    use_text_in_target: bool = False
    alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def process_single_choice(
        self, choice: str, index: int, use_text: bool = True
    ) -> str:
        try:
            processed_choice = f"{self.alphabet[index]}"
        except IndexError as e:
            raise ValueError(
                f"Too many choices, the length of alphabet '{self.alphabet}': {len(self.alphabet)} is the limit"
            ) from e
        if use_text:
            processed_choice += f"{self.enumeration_suffix}{choice}"
        return processed_choice

    def process_choices(self, choices: List[str]) -> str:
        processed_choices = []
        for index, choice in enumerate(choices):
            processed_choices.append(self.process_single_choice(choice, index))
        return self.choices_separator.join(processed_choices)

    def process_target(self, choices, target_index):
        return self.process_single_choice(
            choices[target_index], target_index, use_text=self.use_text_in_target
        )

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        result = super().process(instance, stream_name)
        target_key, target_value = next(iter(result["outputs"].items()))
        choices = result["inputs"][self.choices_field]
        target_index_in_choices = choices.index(target_value)

        processed_choices = self.process_choices(choices)
        processed_target = self.process_target(choices, target_index_in_choices)

        result["inputs"][self.choices_field] = processed_choices
        result["outputs"][target_key] = processed_target

        return result
