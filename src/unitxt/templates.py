import json
from abc import abstractmethod
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .artifact import Artifact
from .collections import ListCollection
from .dataclass import NonPositionalField
from .operator import InstanceOperator
from .random_utils import new_random_generator
from .type_utils import isoftype


class TemplateFormatKeyError(KeyError):
    def __init__(self, template, data, data_type, format_str, format_name):
        keys = ", ".join(data.keys())
        super().__init__(
            f"Available {data_type}s are [{keys}] "
            f"but {template.__class__.__name__}.{format_name} format requires a different ones: '{format_str}'"
        )


class Template(InstanceOperator):
    """The role of template is to take the fields of every instance and verbalize it.

    Meaning the template is taking the instance and generating source, target and references.

    Args:
        skip_rendered_instance (bool): if "source", "target", and "references" are already defined fields in the instance, skip its processing
        postprocessors: a list of strings being artifact names of text processors, to be applied on the model output
        instruction: a formatting string that yields an instruction with potential participation of values from the "inputs" part of the instance
        target_prefix: a string to be used to format the prompt. Not a formatting string.

    """

    skip_rendered_instance: bool = NonPositionalField(default=True)
    postprocessors: List[str] = NonPositionalField(
        default_factory=lambda: ["processors.to_string_stripped"]
    )
    instruction: str = NonPositionalField(default="")
    target_prefix: str = NonPositionalField(default="")
    title_fields: List[str] = NonPositionalField(default_factory=list)

    def inputs_to_instruction_and_target_prefix(self, inputs):
        instruction = self.apply_formatting(
            inputs, "input", self.instruction, "instruction", serialize=True
        )
        target_prefix = self.apply_formatting(
            inputs, "input", self.target_prefix, "target_prefix", serialize=True
        )
        return instruction, target_prefix

    def preprocess_inputs_and_outputs(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return inputs, outputs

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.skip_rendered_instance:
            if (
                "source" in instance
                and "target" in instance
                and "references" in instance
            ):
                return instance

        inputs = instance.get("inputs")
        outputs = instance.get("outputs")
        inputs, outputs = self.preprocess_inputs_and_outputs(inputs, outputs)

        self.set_titles(inputs)
        source = self.inputs_to_source(inputs)
        instruction, target_prefix = self.inputs_to_instruction_and_target_prefix(
            inputs
        )
        target, references = self.outputs_to_target_and_references(outputs)

        return {
            **instance,
            "source": source,
            "target": target,
            "references": references,
            "instruction": instruction,
            "target_prefix": target_prefix,
        }

    @abstractmethod
    def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        pass

    def set_titles(self, data):
        for field in self.title_fields:
            data[field] = data[field].title()

    @abstractmethod
    def outputs_to_target_and_references(
        self, outputs: Dict[str, object]
    ) -> Tuple[str, List[str]]:
        pass

    def get_postprocessors(self) -> List[str]:
        return self.postprocessors

    def serialize_data(self, data):
        return {
            k: ", ".join(str(t) for t in v) if isinstance(v, list) else v
            for k, v in data.items()
        }

    def apply_formatting(
        self, data, data_type, format_str, format_name, serialize=False
    ) -> str:
        if serialize:
            data = self.serialize_data(data)
        try:
            return format_str.format(**data)
        except KeyError as e:
            raise TemplateFormatKeyError(
                self, data, data_type, format_str, format_name
            ) from e


class InputOutputTemplate(Template):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify the formatting strings with which to glue together the input and output designated fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    input_format: str
    output_format: str = None

    def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        return self.apply_formatting(
            inputs, "input", self.input_format, "input_format", serialize=True
        )

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = self.apply_formatting(
            outputs, "output", self.output_format, "output_format", serialize=True
        )
        references = [target]
        return target, references


class InputOutputTemplateWithCustomTarget(InputOutputTemplate):
    reference: str

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = self.apply_formatting(
            outputs, "output", self.output_format, "output_format", serialize=True
        )
        reference = self.apply_formatting(
            outputs, "output", self.reference, "reference", serialize=True
        )
        return target, [reference]


class PairwiseChoiceTemplate(InputOutputTemplate):
    """PairwiseChoiceTemplate.

    Requirements:
     The answer field value should be of type Literal["choice_a", "choice_b", "tie"]

    Args:
         choice_a_field (str): The field which contains choice_a value
         choice_b_field (str): The field which contains choice_b value
         answer_field (str): The field which contains the answer value.
           Should be of type Literal["choice_1", "choice_2", "tie"]
         choice_a_label (str): The label of choice A answer as it is verbalized in the template.
         choice_b_label (str): The label of choice B answer as it is verbalized in the template.
         choice_tie_label (str): The label of a tie answer as it should be verbalized in the template.
         shuffle (bool): whether to shuffle the choices or not. This is done to take into account position bias.

    shuffle: 50% of the time:
     1) The values of choice_a_field and choice_b_field will be swapped.
     2) If the values of answer_field is choice_a_label, set it to choice_b_label.
         Else if the values of answer_field is choice_b_label, set it to choice_a_label.
         Else if the value of answer_field is choice_tie_label, do nothing.

    """

    choice_a_field: str
    choice_b_field: str
    answer_field: str
    choice_a_label: str
    choice_b_label: str
    choice_tie_label: str
    shuffle: bool

    def verbalize_answer_field(self, outputs: Dict[str, object]):
        answer = outputs[self.answer_field]
        assert answer in ["choice_a", "choice_b", "tie"]
        if answer == "choice_a":
            outputs[self.answer_field] = self.choice_a_label
        elif answer == "choice_b":
            outputs[self.answer_field] = self.choice_b_label
        else:
            outputs[self.answer_field] = self.choice_tie_label

        return outputs

    def shuffle_values(self, inputs: Dict[str, object], outputs: Dict[str, object]):
        outcome = random()  # A float between 0 and 1
        if outcome <= 0.5:
            choice_a_value = inputs[self.choice_a_field]
            choice_b_value = inputs[self.choice_b_field]

            inputs[self.choice_a_field] = choice_a_value
            inputs[self.choice_b_field] = choice_b_value

            answer = outputs[self.answer_field]
            assert answer in [
                self.choice_a_label,
                self.choice_b_label,
                self.choice_tie_label,
            ]
            if answer == self.choice_a_label:
                outputs[self.answer_field] = self.choice_b_label
            elif answer == self.choice_b_label:
                outputs[self.answer_field] = self.choice_a_label

        return inputs, outputs

    def preprocess_inputs_and_outputs(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        outputs = self.verbalize_answer_field(outputs)
        inputs, outputs = self.shuffle_values(inputs, outputs)
        return inputs, outputs


class DialogFieldsData(Artifact):
    user_role_label: str
    assistant_role_label: str
    system_role_label: str
    dialog_field: str


class DialogTemplate(InputOutputTemplate):
    dialog_fields: List[DialogFieldsData]
    turns_separator: str = "\n\n"
    label_separator: str = " "

    def process_dialog(self, inputs: Dict[str, object]):
        for dialog_fields in self.dialog_fields:
            dialog = inputs[dialog_fields.dialog_field]
            # TODO: update isoftype method to support Literal verification and check
            #  it's List[Tuple[Literal["user", "assistant", "system"], str]] (Issue #799)
            assert isoftype(dialog, List[Tuple[str, str]])

            user_role_label = dialog_fields.user_role_label
            assistant_role_label = dialog_fields.assistant_role_label
            system_role_label = dialog_fields.system_role_label

            dialog_str = ""
            for i, turn in enumerate(dialog):
                (turn_type, turn_text) = turn
                turns_separator = "" if i == 0 else self.turns_separator
                if turn_type == "user":
                    dialog_str += f"{turns_separator}{user_role_label}{self.label_separator}{turn_text}"
                elif turn_type == "assistant":
                    dialog_str += f"{turns_separator}{assistant_role_label}{self.label_separator}{turn_text}"
                elif turn_type == "system":
                    dialog_str += f"{turns_separator}{system_role_label}{self.label_separator}{turn_text}"

            inputs[dialog_fields.dialog_field] = dialog_str
        return inputs

    def preprocess_inputs_and_outputs(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.process_dialog(inputs), outputs


class DialogPairwiseChoiceTemplate(DialogTemplate, PairwiseChoiceTemplate):
    def preprocess_inputs_and_outputs(
        self, inputs: Dict[str, Any], outputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, outputs = DialogTemplate.preprocess_inputs_and_outputs(
            self, inputs, outputs
        )
        return PairwiseChoiceTemplate.preprocess_inputs_and_outputs(
            self, inputs, outputs
        )


class MultipleChoiceTemplate(Template):
    """Formats the input (that specifies the question), the multiple choices to select the answer from, and specifies the field with the correct answer."""

    input_format: str
    target_prefix: str = ""
    choices_field: str = "choices"
    target_field: str = "label"
    choices_separator: str = ", "
    source_choice_format: str = "{choice_numeral}. {choice_text}"
    target_choice_format: str = "{choice_numeral}"
    enumerator: str = "capitals"
    shuffle_choices: bool = False

    def prepare(self):
        super().prepare()
        if self.enumerator == "capitals":
            self.enumerator = "ABCDEFGHIJKLMNOP"
        if self.enumerator == "lowercase":
            self.enumerator = "abcdefghijklmnop"
        if self.enumerator == "numbers":
            self.enumerator = [str(i + 1) for i in range(20)]
        if self.enumerator == "roman":
            self.enumerator = [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
                "XI",
                "XII",
                "XIII",
                "XIV",
                "XV",
                "XVI",
                "XVII",
                "XVIII",
                "XIX",
                "XX",
            ]

    def inputs_to_choices(self, data: Dict[str, object], choice_format: str) -> str:
        choices = data[self.choices_field]
        enumrated_choices = []
        for i, choice in enumerate(choices):
            enumrated_choices.append(
                choice_format.format(
                    choice_text=choice,
                    choice_numeral=self.enumerator[i],
                )
            )
        return enumrated_choices

    def inputs_to_numerals(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        return self.inputs_to_choices(inputs, "{choice_numeral}")

    def prepare_multiple_choice_inputs(
        self, inputs: Dict[str, object]
    ) -> Dict[str, object]:
        choices = self.inputs_to_choices(inputs, self.source_choice_format)
        return {
            "numerals": self.inputs_to_numerals(inputs),
            **inputs,
            self.choices_field: self.choices_separator.join(choices),
        }

    def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        inputs = self.prepare_multiple_choice_inputs(inputs)
        return self.apply_formatting(
            inputs, "input", self.input_format, "input_format", serialize=True
        )

    def inputs_to_instruction_and_target_prefix(self, inputs):
        inputs = self.prepare_multiple_choice_inputs(inputs)
        return super().inputs_to_instruction_and_target_prefix(inputs)

    def outputs_to_target_index(self, outputs: Dict[str, object]) -> str:
        target = outputs[self.target_field]

        if not isinstance(target, int):
            try:
                return outputs[self.choices_field].index(target)
            except ValueError as e:
                raise ValueError(
                    f"MultipleChoiceTemplate could not locate textual target '{target}' in choices list: {outputs[self.choices_field]}"
                ) from e
        return target

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = outputs[self.target_field]

        if not isinstance(target, int):
            try:
                target = outputs[self.choices_field].index(target)
            except ValueError as e:
                raise ValueError(
                    f"MultipleChoiceTemplate could not locate textual target '{target}' in choices list: {outputs[self.choices_field]}"
                ) from e

        choices = self.inputs_to_choices(outputs, self.target_choice_format)

        try:
            target = choices[target]
        except IndexError as e:
            raise IndexError(
                f"MultipleChoiceTemplate cannot find index number {target} in choices: {choices}"
            ) from e

        return target, [target]

    def _shuffle_choices(self, instance):
        target_index = self.outputs_to_target_index(instance["outputs"])
        original_label_choice = instance["outputs"][self.choices_field][target_index]
        choices = instance["inputs"][self.choices_field]
        random_generator = new_random_generator(
            {**instance["inputs"], **instance["outputs"]}
        )
        random_generator.shuffle(choices)
        instance["inputs"][self.choices_field] = choices
        instance["outputs"][self.choices_field] = choices
        instance["outputs"][self.target_field] = choices.index(original_label_choice)
        return instance

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.shuffle_choices:
            instance = self._shuffle_choices(instance)
        result = super().process(instance, stream_name)
        if "options" not in result["outputs"]:
            result["outputs"]["options"] = self.inputs_to_choices(
                instance["outputs"], self.target_choice_format
            )
        return result


class YesNoTemplate(Template):
    """A template for generating binary Yes/No questions asking whether an input text is of a specific class.

    input_format:
        Defines the format of the question.
    class_field:
        Defines the field that contains the name of the class that this template
        asks of.
    label_field:
        Defines the field which contains the true label of the input text. If a gold label is equal to the
        value in class_name, then the correct output is self.yes_answer (by default, "Yes").
        Otherwise the correct output is self.no_answer (by default, "No").
    yes_answer:
        The output value for when the gold label equals self.class_name.
        Defaults to "Yes".
    no_answer:
        The output value for when the gold label differs from self.class_name.
        Defaults to "No".
    """

    input_format: str = None
    class_field: str = None
    label_field: str = None
    yes_answer: str = "Yes"
    no_answer: str = "No"

    def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        return self.apply_formatting(
            inputs, "input", self.input_format, "input_format", serialize=True
        )

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        try:
            gold_class_names = outputs[self.label_field]
        except KeyError as e:
            raise RuntimeError(
                f"Available outputs are {list(outputs.keys())}, missing required label field: '{self.label_field}'."
            ) from e
        if not isinstance(gold_class_names, list):
            raise RuntimeError(
                f"Unexpected value for gold_class_names: '{gold_class_names}'. Expecting a list."
            )
        try:
            queried_class_name = outputs[self.class_field]
        except KeyError as e:
            raise RuntimeError(
                f"Available outputs are {list(outputs.keys())}, missing required class field: '{self.class_field}'."
            ) from e
        if not queried_class_name or not isinstance(queried_class_name, str):
            raise RuntimeError(
                f"Unexpected value for queried_class_names: '{queried_class_name}'. Expected a string."
            )
        if queried_class_name in gold_class_names:
            return self.yes_answer, [self.yes_answer]
        return self.no_answer, [self.no_answer]


class KeyValTemplate(Template):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify with what separators to glue together the input and output designated fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    pairs_separator: str = ", "
    key_val_separator: str = ": "
    use_keys_for_inputs: bool = True
    outputs_key_val_separator: str = ": "
    use_keys_for_outputs: bool = False

    def process_dict(
        self, data: Dict[str, object], key_val_sep, pairs_sep, use_keys
    ) -> str:
        data = self.serialize_data(data)
        pairs = []
        for key, val in data.items():
            key_val = [key, str(val)] if use_keys else [str(val)]
            pairs.append(key_val_sep.join(key_val))
        return pairs_sep.join(pairs)

    def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
        return self.process_dict(
            inputs,
            key_val_sep=self.key_val_separator,
            pairs_sep=self.pairs_separator,
            use_keys=self.use_keys_for_inputs,
        )

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = self.process_dict(
            outputs,
            key_val_sep=self.key_val_separator,
            pairs_sep=self.pairs_separator,
            use_keys=self.use_keys_for_outputs,
        )
        return target, [target]


class OutputQuantizingTemplate(InputOutputTemplate):
    quantum: Union[float, int] = 0.1  # Now supports both int and float

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        if isinstance(self.quantum, int):
            # When quantum is an int, format quantized values as ints
            quantized_outputs = {
                key: f"{int(round(value / self.quantum) * self.quantum)}"
                for key, value in outputs.items()
            }
        else:
            # When quantum is a float, format quantized values with precision based on quantum
            quantum_str = f"{self.quantum:.10f}".rstrip("0").rstrip(".")
            quantized_outputs = {
                key: f"{round(value / self.quantum) * self.quantum:{quantum_str}}"
                for key, value in outputs.items()
            }
        return super().outputs_to_target_and_references(quantized_outputs)


class MultiLabelTemplate(InputOutputTemplate):
    labels_field: str = "labels"
    labels_separator: str = ", "
    postprocessors: List[str] = ["processors.to_list_by_comma"]
    output_format: str = "{labels}"
    empty_label: str = "None"

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        labels = outputs[self.labels_field]
        if not isinstance(labels, list):
            raise ValueError(
                f"MultiLabelTemplate requires labels field '{self.labels_field}' to be a list. Got {self.labels_field}<{type(labels).__name__}>: {labels}"
            )
        if len(labels) == 0:
            labels = [self.empty_label]
        labels_str = self.labels_separator.join(labels)
        return super().outputs_to_target_and_references({self.labels_field: labels_str})


class MultiReferenceTemplate(InputOutputTemplate):
    references_field: str = "references"
    random_reference: bool = False

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> List[str]:
        references = outputs[self.references_field]
        if not isoftype(references, List[str]):
            raise ValueError(
                f"MultiReferenceTemplate requires references field '{self.references_field}' to be List[str]. Got {self.references_field}<{type(references).__name__}>: {references}"
            )
        if len(references) == 0:
            raise ValueError(
                "No references found. MultiReferenceTemplate requires at least one reference."
            )

        if self.random_reference:
            random_generator = new_random_generator(outputs)
            target = random_generator.choice(references)
        else:
            target = references[0]

        return target, references


def escape_chars(s, chars_to_escape):
    for char in chars_to_escape:
        s = s.replace(char, f"\\{char}")
    return s


class SpanLabelingBaseTemplate(MultiLabelTemplate):
    spans_starts_field: str = "spans_starts"
    spans_ends_field: str = "spans_ends"
    text_field: str = "text"
    labels_support: list = None

    def extract_span_label_pairs(self, outputs):
        spans_starts = outputs[self.spans_starts_field]
        spans_ends = outputs[self.spans_ends_field]
        text = outputs[self.text_field]
        labels = outputs[self.labels_field]

        spans = []
        for span_start, span_end, label in zip(spans_starts, spans_ends, labels):
            if self.labels_support is None or label in self.labels_support:
                spans.append((span_start, span_end, text[span_start:span_end], label))

        for span in sorted(spans):
            if self.labels_support is None or span[3] in self.labels_support:
                yield span[2], span[3]

    def outputs_to_target_and_references(
        self, outputs: Dict[str, object]
    ) -> Dict[str, object]:
        span_labels_pairs = self.extract_span_label_pairs(outputs)
        targets = self.span_label_pairs_to_targets(span_labels_pairs)
        return super().outputs_to_target_and_references({"labels": targets})

    @abstractmethod
    def span_label_pairs_to_targets(self, pairs):
        pass


class SpanLabelingTemplate(SpanLabelingBaseTemplate):
    span_label_format: str = "{span}: {label}"
    escape_characters: List[str] = [":", ","]
    postprocessors: List[str] = ["processors.to_span_label_pairs"]

    def span_label_pairs_to_targets(self, span_label_pairs):
        targets = []
        for span, label in span_label_pairs:
            if self.escape_characters is not None:
                span = escape_chars(span, self.escape_characters)
            target = self.span_label_format.format(span=span, label=label)
            targets.append(target)
        return targets


class SpanLabelingJsonTemplate(SpanLabelingBaseTemplate):
    postprocessors = [
        "processors.load_json",
        "processors.dict_of_lists_to_value_key_pairs",
    ]

    def span_label_pairs_to_targets(self, span_label_pairs):
        groups = {}
        for span, label in span_label_pairs:
            if label not in groups:
                groups[label] = []
            groups[label].append(span)
        if len(groups) > 0:
            targets = [json.dumps(groups, ensure_ascii=False)]
        else:
            targets = []
        return targets


class TemplatesList(ListCollection):
    def verify(self):
        for template in self.items:
            assert isinstance(template, Template)


class TemplatesDict(Dict):
    def verify(self):
        for _key, template in self.items():
            assert isinstance(template, Template)
