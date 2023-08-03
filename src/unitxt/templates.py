from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Dict, List, Union

from .artifact import Artifact
from .instructions import Instruction, TextualInstruction
from .operator import InstanceOperatorWithGlobalAccess, StreamInstanceOperator
from .random_utils import random
from .text_utils import split_words


class Renderer(ABC):
    @abstractmethod
    def get_postprocessors(self) -> List[str]:
        pass


class Template(Artifact):
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

        key, targets = next(iter(outputs.items()))
        if not isinstance(targets, list):
            targets = [targets]

        references = [self.template.process_outputs({key: target}) for target in targets]

        if self.random_reference:
            target = random.choice(references)
        else:
            if len(references) == 0:
                raise ValueError("No references found")
            target = references[0]  # what

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
        elif isinstance(self.template, InputOutputTemplate):
            self.template = AutoInputOutputTemplate(
                input_format=self.template.input_format,
                output_format=self.template.output_format,
            )
        else:
            raise ValueError(
                f"Template must be an instance of InputOutputTemplate or AutoInputOutputTemplate, got {type(self.template)}"
            )

    def render(self, instance: Dict[str, object]) -> Dict[str, object]:
        if not self.template.is_complete():
            self.template.infer_missing(instance["inputs"], instance["outputs"])

        inputs = {key: value for key, value in instance["inputs"].items()}

        return super().render({**instance, "inputs": inputs})


class CharacterSizeLimiter(Artifact):
    limit: int = 1000

    def check(self, text: str) -> bool:
        return len(text) <= self.limit


class RenderTemplatedICL(RenderAutoFormatTemplate):
    instruction: Instruction = None
    input_prefix: str = "Input: "
    output_prefix: str = "Output: "
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


class SpanLabelingTemplate(InputOutputTemplate):
    spans_starts_field: str = "spans_starts"
    spans_ends_field: str = "spans_ends"
    text_field: str = "text"
    labels_field: str = "labels"
    span_label_format: str = "{span}: {label}"
    postprocessors = ["processors.extract_pairs"]

    def process_outputs(self, outputs: Dict[str, object]) -> Dict[str, object]:
        spans_starts = outputs[self.spans_starts_field]
        spans_ends = outputs[self.spans_ends_field]
        text = outputs[self.text_field]
        labels = outputs[self.labels_field]

        spans = []
        for span_start, span_end, label in zip(spans_starts, spans_ends, labels):
            spans.append((span_start, span_end, label))

        spans.sort(key=lambda span: span[0])

        text_spans = []
        for span in spans:
            text_spans.append(text[span[0] : span[1]])

        targets = []
        for span, label in zip(text_spans, labels):
            targets.append(self.span_label_format.format(span=span, label=label))

        return super().process_outputs({"spans_and_labels": targets})


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
