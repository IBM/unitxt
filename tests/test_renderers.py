import unittest

from src.unitxt.formats import ICLFormat
from src.unitxt.instructions import TextualInstruction
from src.unitxt.renderers import (
    RenderDemonstrations,
    RenderFormat,
    RenderInstruction,
    StandardRenderer,
)
from src.unitxt.templates import InputOutputTemplate, MultiReferenceTemplate
from src.unitxt.test_utils.operators import check_operator

template = InputOutputTemplate(
    input_format='This is my sentence: "{text}"', output_format="{label}"
)
instruction = TextualInstruction(
    "classify user sentence by its sentiment to either positive, or negative."
)
format = ICLFormat(
    input_prefix="User:", output_prefix="Agent:", instruction_prefix="Instruction:"
)


class TestRenderers(unittest.TestCase):
    def test_render_template(self):
        instance = {"inputs": {"text": "was so bad"}, "outputs": {"label": "negative"}}

        result = template.process(instance)
        target = {
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_multi_reference_template(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instance = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
        }

        result = template.process(instance)
        target = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
            "source": "This is my sentence: who was he?",
            "target": "Dan",
            "references": ["Dan", "Yossi"],
        }
        self.assertDictEqual(result, target)

    def test_render_demonstrations(self):
        renderer = RenderDemonstrations(template=template, demos_field="demos")

        instance = {
            "demos": [
                {
                    "inputs": {"text": "was so not good"},
                    "outputs": {"label": "negative"},
                },
                {"inputs": {"text": "was so good"}, "outputs": {"label": "positive"}},
            ]
        }

        result = renderer.process(instance)

        target = {
            "demos": [
                {
                    "inputs": {"text": "was so not good"},
                    "outputs": {"label": "negative"},
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "inputs": {"text": "was so good"},
                    "outputs": {"label": "positive"},
                    "source": 'This is my sentence: "was so good"',
                    "target": "positive",
                    "references": ["positive"],
                },
            ]
        }

        self.assertDictEqual(result, target)

    def test_render_demonstrations_multi_reference(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        renderer = RenderDemonstrations(template=template, demos_field="demos")

        instance = {
            "demos": [
                {
                    "inputs": {"text": "who was he?"},
                    "outputs": {"answer": ["Dan", "Yossi"]},
                },
                {
                    "inputs": {"text": "who was she?"},
                    "outputs": {"answer": ["Shira", "Yael"]},
                },
            ]
        }

        result = renderer.process(instance)

        target = {
            "demos": [
                {
                    "inputs": {"text": "who was he?"},
                    "outputs": {"answer": ["Dan", "Yossi"]},
                    "source": "This is my sentence: who was he?",
                    "target": "Dan",
                    "references": ["Dan", "Yossi"],
                },
                {
                    "inputs": {"text": "who was she?"},
                    "outputs": {"answer": ["Shira", "Yael"]},
                    "source": "This is my sentence: who was she?",
                    "target": "Shira",
                    "references": ["Shira", "Yael"],
                },
            ]
        }

        self.assertDictEqual(result, target)

    def test_render_instruction(self):
        renderer = RenderInstruction(instruction=instruction)

        instance = {}
        result = renderer.process(instance)
        target = {
            "instruction": "classify user sentence by its sentiment to either positive, or negative."
        }
        self.assertDictEqual(result, target)

    def test_render_format(self):
        renderer = RenderFormat(format=format, demos_field="demos")

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
                    "target": "positive",
                    "references": ["positive"],
                },
            ],
        }

        result = renderer.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so not good"\nAgent: negative\n\nUser:This is my sentence: "was so good"\nAgent: positive\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_format_no_demos(self):
        renderer = RenderFormat(format=format)

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
        }

        result = renderer.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_render_format_with_prefix_and_suffix(self):
        format_fix = ICLFormat(
            input_prefix="User: ",
            output_prefix="Agent: ",
            target_prefix="",
            prefix="[INST] <<SYS>>\n",
            suffix="[/INST]",
        )
        renderer = RenderFormat(format=format_fix, demos_field="demos")

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
            "demos": [
                {
                    "source": 'This is my sentence: "was so not good"',
                    "target": "negative",
                    "references": ["negative"],
                },
                {
                    "source": 'This is my sentence: "was so good"',
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
        renderer = StandardRenderer(
            template=template,
            instruction=instruction,
            format=format,
            demos_field="demos",
        )

        instance = {
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "demos": [
                {
                    "inputs": {"text": "was so not good"},
                    "outputs": {"label": "negative"},
                },
                {"inputs": {"text": "was so good"}, "outputs": {"label": "positive"}},
            ],
        }

        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so not good"\nAgent: negative\n\nUser:This is my sentence: "was so good"\nAgent: positive\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
        }

        check_operator(
            operator=renderer, inputs=[instance], targets=[target], tester=self
        )

    def test_standard_renderer_multi_reference(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instruction = TextualInstruction("answer the question")

        renderer = StandardRenderer(
            template=template,
            instruction=instruction,
            format=format,
            demos_field="demos",
        )

        instance = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
            "demos": [
                {
                    "inputs": {"text": "who was she?"},
                    "outputs": {"answer": ["Shira", "Yael"]},
                },
                {
                    "inputs": {"text": "who was he?"},
                    "outputs": {"answer": ["Codi", "Bodi"]},
                },
            ],
        }

        target = {
            "source": "Instruction:answer the question\n\nUser:This is my sentence: who was she?\nAgent: Shira\n\nUser:This is my sentence: who was he?\nAgent: Codi\n\nUser:This is my sentence: who was he?\nAgent:",
            "target": "Dan",
            "references": ["Dan", "Yossi"],
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
        }

        check_operator(
            operator=renderer, inputs=[instance], targets=[target], tester=self
        )

    def test_temp(self):
        from src.unitxt.blocks import (
            AddFields,
            FormTask,
            InputOutputTemplate,
            LoadHF,
            MapInstanceValues,
            TaskCard,
            TemplatesList,
        )
        from src.unitxt.catalog import add_to_catalog
        from src.unitxt.test_utils.card import test_card

        card = TaskCard(
            loader=LoadHF(path="glue", name="cola"),
            preprocess_steps=[
                "splitters.small_no_test",
                MapInstanceValues(
                    mappers={"label": {"0": "unacceptable", "1": "acceptable"}}
                ),
                AddFields(
                    fields={
                        "choices": ["unacceptable", "acceptable"],
                    }
                ),
            ],
            task=FormTask(
                inputs=["choices", "sentence"],
                outputs=["label"],
                metrics=["metrics.matthews_correlation"],
            ),
            templates=TemplatesList(
                [
                    InputOutputTemplate(
                        input_format="""
                            Given this sentence: {sentence}, classify if it is {choices}.
                        """.strip(),
                        output_format="{label}",
                    ),
                ]
            ),
        )

        test_card(card, strict=False)
        add_to_catalog(card, "cards.cola", overwrite=True)
