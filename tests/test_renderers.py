import unittest

from src.unitxt.instructions import TextualInstruction
from src.unitxt.operators import AddFields, ModelInputFormatter
from src.unitxt.renderers import (
    RenderDemonstrations,
)
from src.unitxt.templates import InputOutputTemplate, MultiReferenceTemplate

template = InputOutputTemplate(
    input_format='This is my sentence: "{text}"', output_format="{label}"
)
instruction = TextualInstruction(
    "classify user sentence by its sentiment to either positive, or negative."
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
        instruction_adder = AddFields(
            fields={"instruction": instruction() if instruction is not None else ""}
        )

        instance = {}
        result = instruction_adder.process(instance)
        target = {
            "instruction": "classify user sentence by its sentiment to either positive, or negative."
        }
        self.assertDictEqual(result, target)

    def test_model_input_formatter(self):
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

        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )

        result = model_input_formatter.process(instance)

        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so not good"\nAgent:negative\n\nUser:This is my sentence: "was so good"\nAgent:positive\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_model_input_formatter_no_demos(self):
        instance = {
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "classify user sentence by its sentiment to either positive, or negative.",
        }
        model_input_formatter = ModelInputFormatter(
            demo_format="User:{source}\nAgent:{target}\n\n",
            model_input_format="Instruction:{instruction}\n\n{demos}User:{source}\nAgent:",
        )
        result = model_input_formatter.process(instance)
        target = {
            "source": 'Instruction:classify user sentence by its sentiment to either positive, or negative.\n\nUser:This is my sentence: "was so bad"\nAgent:',
            "target": "negative",
            "references": ["negative"],
        }
        self.assertDictEqual(result, target)

    def test_model_input_formatter_with_prefix_and_suffix(self):
        model_input_formatter_fix = ModelInputFormatter(
            demos_field="demos",
            demo_format="User: {source}\nAgent: {target}\n\n",
            model_input_format="[INST] <<SYS>>\n{instruction}\n\n{demos}User: {source}\nAgent: [/INST]",
        )
        renderer = model_input_formatter_fix

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
