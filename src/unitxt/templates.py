import json
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Dict, List, Optional, Union

from .artifact import Artifact
from .dataclass import NonPositionalField
from .instructions import Instruction, TextualInstruction
from .operator import InstanceOperatorWithGlobalAccess, StreamInstanceOperator
from .random_utils import random
from .text_utils import split_words


class Renderer(ABC):
    @abstractmethod
    def get_postprocessors(self) -> List[str]:
        pass


class Template(Artifact):
    is_multi_target: bool = NonPositionalField(default=False)
    is_multi_reference: bool = NonPositionalField(default=False)

    @abstractmethod
    def process_inputs(self, inputs: Dict[str, object]) -> Dict[str, object]:
        pass

    @abstractmethod
    def process_outputs(self, outputs: Dict[str, object]) -> Dict[str, object]:
        pass

    @abstractmethod
    def get_postprocessors(self) -> List[str]:
        pass


class RenderFormatTemplate(Renderer, StreamInstanceOperator):
    template: Template = None
    random_reference: bool = False

    def verify(self):
        assert isinstance(self.template, Template), "Template must be an instance of Template"
        assert self.template is not None, "Template must be specified"

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        return self.render(instance)

    def render(self, instance: Dict[str, Any]) -> Dict[str, Any]:
        inputs = instance.pop("inputs")
        outputs = instance.pop("outputs")

        source = self.template.process_inputs(inputs)
        targets = self.template.process_outputs(outputs)

        if self.template.is_multi_reference:
            references = targets
            if self.random_reference:
                target = random.choice(references)
            else:
                if len(references) == 0:
                    raise ValueError("No references found")
                target = references[0]
        else:
            references = [targets]
            target = targets

        return {
            **instance,
            "source": source,
            "target": target,
            "references": references,
        }

    def get_postprocessors(self) -> List[str]:
        return self.template.get_postprocessors()


class RenderAutoFormatTemplate(RenderFormatTemplate):
    def prepare(self):
        if self.template is None:
            self.template = AutoInputOutputTemplate()

    def render(self, instance: Dict[str, object]) -> Dict[str, object]:
        try:
            if not self.template.is_complete():
                self.template.infer_missing(instance["inputs"], instance["outputs"])
        except:
            pass

        inputs = {key: value for key, value in instance["inputs"].items()}

        return super().render({**instance, "inputs": inputs})


class CharacterSizeLimiter(Artifact):
    limit: int = 1000

    def check(self, text: str) -> bool:
        return len(text) <= self.limit


class RenderTemplatedICL(RenderAutoFormatTemplate):
    instruction: Instruction = None
    input_prefix: str = "Input: "
    output_prefix: str = "Output:"
    target_prefix: str = " "
    instruction_prefix: str = ""
    demos_field: str = None
    size_limiter: Artifact = None
    input_output_separator: str = "\n"
    demo_separator: str = "\n\n"

    def render(self, instance: Dict[str, object]) -> Dict[str, object]:
        demos = instance.pop(self.demos_field, [])

        source = ""

        example = super().render(instance)

        input_str = self.input_prefix + example["source"] + self.input_output_separator + self.output_prefix

        if self.instruction is not None:
            source += self.instruction_prefix + self.instruction() + self.demo_separator

        for demo_instance in demos:
            demo_example = super().render(demo_instance)
            demo_str = (
                self.input_prefix
                + demo_example["source"]
                + self.input_output_separator
                + self.output_prefix
                + self.target_prefix
                + demo_example["target"]
                + self.demo_separator
            )

            if self.size_limiter is not None:
                if not self.size_limiter.check(source + demo_str + input_str + example["target"]):
                    continue

            source += demo_str

        source += input_str

        return {
            **example,
            "source": source,
        }


class InputOutputTemplate(Template):
    input_format: str = None
    output_format: str = None
    postprocessors: List[str] = field(default_factory=lambda: ["processors.to_string"])

    def process_template(self, template: str, data: Dict[str, object]) -> str:
        data = {k: ", ".join(v) if isinstance(v, list) else v for k, v in data.items()}
        return template.format(**data)

    def process_inputs(self, inputs: Dict[str, object]) -> str:
        try:
            return self.process_template(self.input_format, inputs)
        except KeyError as e:
            raise KeyError(
                f"Available inputs are {inputs.keys()} but input format requires a different one: {self.input_format}"
            )

    def process_outputs(self, outputs: Dict[str, object]) -> str:
        try:
            return self.process_template(self.output_format, outputs)
        except KeyError as e:
            raise KeyError(
                f"Available inputs are {outputs.keys()} but output format requires a different one: {self.output_format}"
            )

    def get_postprocessors(self) -> List[str]:
        return self.postprocessors


class OutputQuantizingTemplate(InputOutputTemplate):
    quantum: float = 0.1

    def process_outputs(self, outputs: Dict[str, object]) -> Dict[str, object]:
        quantized_outputs = {
            key: round(input_float / self.quantum) * self.quantum for key, input_float in outputs.items()
        }
        return super().process_outputs(quantized_outputs)


class MultiLabelTemplate(InputOutputTemplate):
    labels_field: str = "labels"
    labels_seprator: str = ", "
    postprocessors = ["processors.to_list"]
    output_format = "{labels}"
    empty_label = "None"

    def process_outputs(self, outputs: Dict[str, object]) -> Dict[str, object]:
        labels = outputs[self.labels_field]
        if len(labels) == 0:
            labels = [self.empty_label]
        labels_str = self.labels_seprator.join(labels)
        return super().process_outputs({"labels": labels_str})


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

    def process_outputs(self, outputs: Dict[str, object]) -> Dict[str, object]:
        span_lables_pairs = self.extract_span_label_pairs(outputs)
        targets = self.span_label_pairs_to_targets(span_lables_pairs)
        return super().process_outputs({"labels": targets})

    @abstractmethod
    def span_label_pairs_to_targets(self, pairs):
        pass


class SpanLabelingTemplate(SpanLabelingBaseTemplate):
    span_label_format: str = "{span}: {label}"
    escape_characters: List[str] = [":", ","]
    postprocessors = ["processors.to_span_label_pairs"]

    def span_label_pairs_to_targets(self, span_label_pairs):
        targets = []
        for span, label in span_label_pairs:
            if self.escape_characters is not None:
                span = escape_chars(span, self.escape_characters)
            target = self.span_label_format.format(span=span, label=label)
            targets.append(target)
        return targets


class SpanLabelingJsonTemplate(SpanLabelingBaseTemplate):
    postprocessors = ["processors.load_json", "processors.dict_of_lists_to_value_key_pairs"]

    def span_label_pairs_to_targets(self, span_label_pairs):
        groups = {}
        for span, label in span_label_pairs:
            if label not in groups:
                groups[label] = list()
            groups[label].append(span)
        if len(groups) > 0:
            targets = [json.dumps(groups)]
        else:
            targets = []
        return targets


class AutoInputOutputTemplate(InputOutputTemplate):
    def infer_input_format(self, inputs):
        input_format = ""
        for key in inputs.keys():
            name = " ".join(word.lower().capitalize() for word in split_words(key) if word != " ")
            input_format += name + ": " + "{" + key + "}" + "\n"
        self.input_format = input_format

    def infer_output_format(self, outputs):
        self.output_format = "{" + next(iter(outputs.keys())) + "}"

    def infer_missing(self, inputs, outputs):
        if self.input_format is None:
            self.infer_input_format(inputs)
        if self.output_format is None:
            self.infer_output_format(outputs)

    def is_complete(self):
        return self.input_format is not None and self.output_format is not None


from .collections import ListCollection


class TemplatesList(ListCollection):
    def verify(self):
        for template in self.items:
            assert isinstance(template, Template)


def outputs_inputs2templates(inputs: Union[str, List], outputs: Union[str, List]) -> TemplatesList:
    """
    combines input and output formats into their dot product
    :param inputs: list of input formats (or one)
    :param outputs: list of output formats (or one)
    :return: TemplatesList of InputOutputTemplate
    """
    templates = []
    if isinstance(inputs, str):
        inputs = [inputs]
    if isinstance(outputs, str):
        outputs = [outputs]
    for input in inputs:
        for output in outputs:
            templates.append(
                InputOutputTemplate(
                    input_format=input.strip(),
                    output_format=output.strip(),
                ),
            )
    return TemplatesList(templates)


def instructions2templates(
    instructions: List[TextualInstruction], templates: List[InputOutputTemplate]
) -> TemplatesList:
    """
    Insert instructions into per demonstration templates
    :param instructions:
    :param templates: strings containing {instuction} where the instruction should be placed
    :return:
    """
    res_templates = []
    for instruction in instructions:
        for template in templates:
            res_templates.append(
                InputOutputTemplate(
                    input_format=template.input_format.replace("{instruction}", instruction.text),
                    output_format=template.output_format,
                )
            )
    return TemplatesList(templates)


class TemplatesDict(Dict):
    def verify(self):
        for key, template in self.items():
            assert isinstance(template, Template)
