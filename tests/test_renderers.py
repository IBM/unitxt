import unittest

from src.unitxt.formats import ICLFormat
from src.unitxt.instructions import TextualInstruction
from src.unitxt.renderers import (
    RenderDemonstrations,
    RenderFormat,
    RenderInstruction,
    RenderTemplate,
    StandardRenderer,
)
from src.unitxt.templates import CompletionTemplate, InputOutputTemplate
from src.unitxt.test_utils.operators import check_operator

template = InputOutputTemplate(input_format='This is my sentence: "{text}"', output_format="{label}")
completion_template = CompletionTemplate(input_format='The sentence "{text}" is ', output_format="{label}")

instruction = TextualInstruction("classify user sentence by its sentiment to either positive, or negative.")
format = ICLFormat(input_prefix="User: ", output_prefix="Agent: ", instruction_prefix="Instruction:")


class TestRenderers(unittest.TestCase):
    def test_render_template(self):
        renderer = RenderTemplate(template=template)

        instance = {"inputs": {"text": "was so bad"}, "outputs": {"label": "negative"}}

        result = renderer.process(instance)
        target = {
            "source": 'This is my sentence: "was so bad"',
            "input_output_separator": "\n",
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

        renderer = RenderTemplate(template=completion_template)
        instance = {"inputs": {"text": "was so bad"}, "outputs": {"label": "negative"}}

        result = renderer.process(instance)
        target = {
            "source": 'The sentence "was so bad" is ',
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "input_output_separator": "",
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_demonstrations(self):
        renderer = RenderDemonstrations(template=template, demos_field="demos")

        instance = {
            "demos": [
                {"inputs": {"text": "was so not good"}, "outputs": {"label": "negative"}},
                {"inputs": {"text": "was so good"}, "outputs": {"label": "positive"}},
            ]
        }

        result = renderer.process(instance)

        target = {
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "input_output_separator": "\n",
                    "inputs": {"text": "was so not good"},
                    "outputs": {"label": "negative"},
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "input_output_separator": "\n",
                    "inputs": {"text": "was so good"},
                    "outputs": {"label": "positive"},
                    "target": "positive",
                    "references": ["positive"],
                },
            ]
        }

        self.assertDictEqual(result, target)

    def test_render_instruction(self):
        renderer = RenderInstruction(instruction=instruction)

        instance = {}
        result = renderer.process(instance)
        target = {"instruction": "classify user sentence by its sentiment to either positive, or negative."}
        self.assertDictEqual(result, target)

    def test_render_format(self):
        renderer = RenderFormat(format=format, demos_field="demos")

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "input_output_separator": "\n",
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or nagative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "input_output_separator": "\n",
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "input_output_separator": "\n",
                    "target": "positive",
                    "references": ["positive"],
                },
            ],
        }

        result = renderer.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or nagative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: ',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_format_no_demos(self):
        renderer = RenderFormat(format=format)

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "input_output_separator": "\n",
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or nagative.",
        }

        result = renderer.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or nagative.\n\nUser: This is my sentence: "was so bad"\nAgent: ',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_format_with_prefix_and_suffix(self):
        format_fix = ICLFormat(
            input_prefix="User: ",
            input_output_separator="\n",
            output_prefix="Agent: ",
            prefix="[INST] <<SYS>>\n",
            suffix="[/INST]",
        )
        renderer = RenderFormat(format=format_fix, demos_field="demos")

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "input_output_separator": "koko",
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "input_output_separator": "koko",
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "input_output_separator": "koko",
                    "target": "positive",
                    "references": ["positive"],
                },
            ],
        }
        self.maxDiff = None
        result = renderer.process(instance)
        target = {
            "source": '[INST] <<SYS>>\nclassify user sentence by its sentiment to either positive, or negative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: [/INST]',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_standard_renderer(self):
        renderer = StandardRenderer(template=template, instruction=instruction, format=format, demos_field="demos")
        self.maxDiff = None
        instance = {
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "demos": [
                {"inputs": {"text": "was so not good"}, "outputs": {"label": "negative"}},
                {"inputs": {"text": "was so good"}, "outputs": {"label": "positive"}},
            ],
        }

        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: ',
            "target": "negative",
            "references": ["negative"],
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
        }

        check_operator(operator=renderer, inputs=[instance], targets=[target], tester=self)
