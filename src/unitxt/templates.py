import json
from abc import abstractmethod
from dataclasses import field
from typing import Any, Dict, List, Optional, Tuple

from .collections import ListCollection
from .dataclass import NonPositionalField
from .operator import StreamInstanceOperator
from .random_utils import new_random_generator
from .type_utils import isoftype


class Template(StreamInstanceOperator):
    """The role of template is to take the fields of every instance and verbalize it.

    Meaning the template is taking the instance and generating source, target and references.
    """

    skip_rendered_instance: bool = NonPositionalField(default=True)
    postprocessors: List[str] = NonPositionalField(
        default_factory=lambda: ["processors.to_string_stripped"]
    )

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

        source = self.inputs_to_source(inputs)
        target, references = self.outputs_to_target_and_references(outputs)

        return {
            **instance,
            "source": source,
            "target": target,
            "references": references,
        }

    @abstractmethod
    def inputs_to_source(self, inputs: Dict[str, object]) -> str:
        pass

    @abstractmethod
    def outputs_to_target_and_references(
        self, outputs: Dict[str, object]
    ) -> Tuple[str, List[str]]:
        pass

    def get_postprocessors(self) -> List[str]:
        return self.postprocessors


class InputOutputTemplate(Template):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify the formatting strings with which to glue together the input and output designated fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    input_format: str = None
    output_format: str = None

    def process_template(self, template: str, data: Dict[str, object]) -> str:
        data = {k: ", ".join(v) if isinstance(v, list) else v for k, v in data.items()}
        return template.format(**data)

    def inputs_to_source(self, inputs: Dict[str, object]) -> str:
        try:
            return self.process_template(self.input_format, inputs)
        except KeyError as e:
            raise KeyError(
                f"Available inputs are {list(inputs.keys())} but input format requires a different ones: '{self.input_format}'"
            ) from e

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        try:
            target = self.process_template(self.output_format, outputs)
        except KeyError as e:
            raise KeyError(
                f"Available outputs are {outputs.keys()} but output format requires a different one: {self.output_format}"
            ) from e

        references = [target]
        return target, references


class MultipleChoiceTemplate(Template):
    """Formats the input (that specifies the question), the multiple choices to select the answer from, and specifies the field with the correct answer."""

    input_format: str
    target_prefix: str = ""
    choices_field: str = "choices"
    target_field: str = "label"
    choices_seperator: str = ", "
    source_choice_format: str = "{choice_numeral}. {choice_text}"
    target_choice_format: str = "{choice_numeral}"
    add_numerals_as_field: str = None
    enumerator: str = "capitals"

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

    def get_choices(self, data: Dict[str, object], choice_format: str) -> str:
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

    def inputs_to_source(self, inputs: Dict[str, object]) -> str:
        choices = self.get_choices(inputs, self.source_choice_format)
        inputs = {
            "numerals": ",".join(self.get_choices(inputs, "{choice_numeral}")),
            **inputs,
            self.choices_field: self.choices_seperator.join(choices),
        }
        try:
            return self.input_format.format(**inputs)
        except KeyError as e:
            raise KeyError(
                f"Available inputs are {inputs.keys()} but input format requires a different one: {self.input_format}"
            ) from e

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = outputs[self.target_field]

        if not isinstance(target, int):
            try:
                target = outputs[self.choices_field].index(target)
            except ValueError as e:
                raise ValueError(
                    f"MultipleChoiceTemplate could not locate textual target '{target}' in choices list: {outputs[self.choices_field]}"
                ) from e

        choices = self.get_choices(outputs, self.target_choice_format)

        try:
            target = choices[target]
        except IndexError as e:
            raise IndexError(
                f"MultipleChoiceTemplate cannot find index number {target} in choices: {choices}"
            ) from e

        return target, [target]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        result = super().process(instance, stream_name)
        if "options" not in result["outputs"]:
            result["outputs"]["options"] = self.get_choices(
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
    postprocessors: List[str] = field(
        default_factory=lambda: ["processors.to_string_stripped"]
    )

    def inputs_to_source(self, inputs: Dict[str, object]) -> str:
        try:
            data = {
                k: ", ".join(v) if isinstance(v, list) else v for k, v in inputs.items()
            }
            return self.input_format.format(**data)
        except KeyError as e:
            raise RuntimeError(
                f"Available inputs are {list(inputs.keys())} but input format requires a different one: {self.input_format}"
            ) from e

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        try:
            gold_class_names = outputs[self.label_field]
        except KeyError as e:
            raise RuntimeError(
                f"Available outputs are {list(outputs.keys())}, missing required label field: '{self.label_field}'."
            ) from e
        if not isinstance(gold_class_names, list) or not gold_class_names:
            raise RuntimeError(
                f"Unexpected value for gold_class_names: '{gold_class_names}'. Expected a non-empty list."
            )
        try:
            queried_class_names = outputs[self.class_field]
        except KeyError as e:
            raise RuntimeError(
                f"Available outputs are {list(outputs.keys())}, missing required class field: '{self.class_field}'."
            ) from e
        if (
            not queried_class_names
            or not isinstance(queried_class_names, list)
            or not len(queried_class_names) == 1
        ):
            raise RuntimeError(
                f"Unexpected value for queried_class_names: '{queried_class_names}'. Expected a list with one item."
            )
        queried_class_name = queried_class_names[0]
        if queried_class_name in gold_class_names:
            return self.yes_answer, [self.yes_answer]
        return self.no_answer, [self.no_answer]

    def get_postprocessors(self) -> List[str]:
        return self.postprocessors


class KeyValTemplate(Template):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify with what separators to glue together the input and output designated fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    pairs_seperator: str = ", "
    key_val_seperator: str = ": "
    use_keys_for_inputs: bool = True
    outputs_key_val_seperator: str = ": "
    use_keys_for_outputs: bool = False

    postprocessors: List[str] = field(
        default_factory=lambda: ["processors.to_string_stripped"]
    )

    def process_dict(
        self, dic: Dict[str, object], key_val_sep, pairs_sep, use_keys
    ) -> str:
        dic = {
            k: ", ".join([str(vi) for vi in v]) if isinstance(v, list) else v
            for k, v in dic.items()
        }
        pairs = []
        for key, val in dic.items():
            key_val = [key, str(val)] if use_keys else [str(val)]
            pairs.append(key_val_sep.join(key_val))
        return pairs_sep.join(pairs)

    def inputs_to_source(self, inputs: Dict[str, object]) -> str:
        return self.process_dict(
            inputs,
            key_val_sep=self.key_val_seperator,
            pairs_sep=self.pairs_seperator,
            use_keys=self.use_keys_for_inputs,
        )

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        target = self.process_dict(
            outputs,
            key_val_sep=self.key_val_seperator,
            pairs_sep=self.pairs_seperator,
            use_keys=self.use_keys_for_outputs,
        )
        return target, [target]

    def get_postprocessors(self) -> List[str]:
        return self.postprocessors


class OutputQuantizingTemplate(InputOutputTemplate):
    quantum: float = 0.1

    def outputs_to_target_and_references(self, outputs: Dict[str, object]) -> str:
        quantum_str = f"{self.quantum:.10f}".rstrip("0").rstrip(".")
        quantized_outputs = {
            key: f"{round(value / self.quantum) * self.quantum:{quantum_str}}"
            for key, value in outputs.items()
        }
        return super().outputs_to_target_and_references(quantized_outputs)


class MultiLabelTemplate(InputOutputTemplate):
    labels_field: str = "labels"
    labels_seprator: str = ", "
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
        labels_str = self.labels_seprator.join(labels)
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
        span_lables_pairs = self.extract_span_label_pairs(outputs)
        targets = self.span_label_pairs_to_targets(span_lables_pairs)
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
