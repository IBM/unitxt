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
from src.unitxt.templates import InputOutputTemplate
from src.unitxt.test_utils.operators import test_operator

template = InputOutputTemplate(input_format='This is my sentence: "{text}"', output_format="{label}")
instruction = TextualInstruction("classify user sentence by its sentiment to either positive, or nagative.")
format = ICLFormat(
    input_prefix="User: ", output_prefix="Agent: ", instruction_prefix="Instruction:", input_output_separator="\n"
)


class TestRenderers(unittest.TestCase):
    def test_render_template(self):
        renderer = RenderTemplate(template=template)

        instance = {"inputs": {"text": "was so bad"}, "outputs": {"label": "negative"}}

        result = renderer.process(instance)
        target = {"source": 'This is my sentence: "was so bad"', "target": "negative", "references": ["negative"]}
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
                {"source": 'This is my sentence: "was so not good"', "target": "negative", "references": ["negative"]},
                {"source": 'This is my sentence: "was so good"', "target": "positive", "references": ["positive"]},
            ]
        }

        self.assertDictEqual(result, target)

    def test_render_instruction(self):
        renderer = RenderInstruction(instruction=instruction)

        instance = {}
        result = renderer.process(instance)
        target = {"instruction": "classify user sentence by its sentiment to either positive, or nagative."}
        self.assertDictEqual(result, target)

    def test_render_format(self):
        renderer = RenderFormat(format=format, demos_field="demos")

        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or nagative.",
            "demos": [
                {"source": 'This is my sentence: "was so not good"', "target": "negative", "references": ["negative"]},
                {"source": 'This is my sentence: "was so good"', "target": "positive", "references": ["positive"]},
            ],
        }

        result = renderer.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or nagative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: ',
            "target": "negative",
            "references": ["negative"],
        }
        from src.unitxt.text_utils import print_dict

        print_dict(result)
        print_dict(target)
        self.assertDictEqual(result, target)

    def test_standard_renderer(self):
        renderer = StandardRenderer(template=template, instruction=instruction, format=format, demos_field="demos")

        instance = {
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "demos": [
                {"inputs": {"text": "was so not good"}, "outputs": {"label": "negative"}},
                {"inputs": {"text": "was so good"}, "outputs": {"label": "positive"}},
            ],
        }

        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or nagative.\n\nUser: This is my sentence: "was so not good"\nAgent: negative\n\nUser: This is my sentence: "was so good"\nAgent: positive\n\nUser: This is my sentence: "was so bad"\nAgent: ',
            "target": "negative",
            "references": ["negative"],
        }

        test_operator(operator=renderer, inputs=[instance], targets=[target], tester=self)

    def test_temp(self):
        import datasets as ds
        from src.unitxt.blocks import (
            AddFields,
            FormTask,
            InputOutputTemplate,
            LoadHF,
            MapInstanceValues,
            NormalizeListFields,
            SplitRandomMix,
            TaskCard,
            TemplatesList,
        )
        from src.unitxt.catalog import add_to_catalog
        from src.unitxt.test_utils.card import test_card

        card = TaskCard(
            loader=LoadHF(path="glue", name="cola"),
            preprocess_steps=[
                "splitters.small_no_test",
                MapInstanceValues(mappers={"label": {"0": "unacceptable", "1": "acceptable"}}),
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
