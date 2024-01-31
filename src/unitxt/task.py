from typing import Any, Dict, List, Optional

from .operator import StreamInstanceOperator


class Tasker:
    pass


class FormTask(Tasker, StreamInstanceOperator):
    """FormTask packs the different instance fields into dictionaries by their roles in the task.

    The output instance contains three fields:
        "inputs" whose value is a sub-dictionary of the input instance, consisting of all the fields listed in Arg 'inputs'.
        "outputs" -- for the fields listed in Arg "outputs".
        "metrics" -- to contain the value of Arg 'metrics'

    """

    inputs: List[str]
    outputs: List[str]
    metrics: List[str]
    augmentable_inputs: List[str] = []

    def verify(self):
        for augmentable_input in self.augmentable_inputs:
            assert (
                augmentable_input in self.inputs
            ), f"augmentable_input f{augmentable_input} is not part of {self.inputs}"

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        try:
            inputs = {key: instance[key] for key in self.inputs}
        except KeyError as e:
            raise KeyError(
                f"Unexpected FormTask input column names ({[key for key in self.inputs if key not in instance]})."
                f"The available input names: {list(instance.keys())}"
            ) from e
        try:
            outputs = {key: instance[key] for key in self.outputs}
        except KeyError as e:
            raise KeyError(
                f"Unexpected FormTask output column names: {[key for key in self.outputs if key not in instance]}"
                f" \n available names:{list(instance.keys())}\n given output names:{self.outputs}"
            ) from e

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
