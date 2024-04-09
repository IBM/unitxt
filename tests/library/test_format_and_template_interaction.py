from unitxt.formats import SystemFormat
from unitxt.operators import SequentialOperator
from unitxt.templates import InputOutputTemplate
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestFormatAndTemplateInteraction(UnitxtTestCase):
    def test_interactions(self):
        instance = {"inputs": {"question": "what?"}, "outputs": {"answer": "that!"}}
        target = "that!"

        template_separated = InputOutputTemplate(
            input_format="Question: {question}",
            target_prefix="Answer: ",
            output_format="{answer}",
        )
        template_combined = InputOutputTemplate(
            input_format="Question:\n{question}\nAnswer:\n",
            output_format="{answer}",
        )

        simple_format = "{source}\n{target_prefix}"
        simple_fixed = "{source}\\N{target_prefix}"
        user_assistant_format = "|user|\n{source}\n|assistant|\n{target_prefix}"
        user_assistant_fixed = "|user|\n{source}\\N|assistant|\n{target_prefix}"
        assistant_last_format = "|user|\n{source}\n{target_prefix}\n|assistant|\n"
        assistant_last_fixed = "|user|\n{source}\\N{target_prefix}\\N|assistant|\n"

        combination_to_required_input = [
            [(template_separated, simple_format), "Question: what?\nAnswer: "],
            [(template_separated, simple_fixed), "Question: what?\nAnswer: "],
            [(template_combined, simple_format), "Question:\nwhat?\nAnswer:\n\n"],
            [(template_combined, simple_fixed), "Question:\nwhat?\nAnswer:\n"],
            [
                (template_separated, user_assistant_format),
                "|user|\nQuestion: what?\n|assistant|\nAnswer: ",
            ],
            [
                (template_separated, user_assistant_fixed),
                "|user|\nQuestion: what?\n|assistant|\nAnswer: ",
            ],
            [
                (template_combined, user_assistant_format),
                "|user|\nQuestion:\nwhat?\nAnswer:\n\n|assistant|\n",
            ],
            [
                (template_combined, user_assistant_fixed),
                "|user|\nQuestion:\nwhat?\nAnswer:\n|assistant|\n",
            ],
            [
                (template_separated, assistant_last_format),
                "|user|\nQuestion: what?\nAnswer: \n|assistant|\n",
            ],
            [
                (template_separated, assistant_last_fixed),
                "|user|\nQuestion: what?\nAnswer: \n|assistant|\n",
            ],
            [
                (template_combined, assistant_last_format),
                "|user|\nQuestion:\nwhat?\nAnswer:\n\n\n|assistant|\n",
            ],
            [
                (template_combined, assistant_last_fixed),
                "|user|\nQuestion:\nwhat?\nAnswer:\n|assistant|\n",
            ],
        ]
        for (template, format), required_input in combination_to_required_input:
            input_format = template.input_format
            target_prefix = template.target_prefix if template.target_prefix else ""

            with self.subTest(
                f"Template: {input_format}, Target Prefix:{target_prefix}, Format: {format}"
            ):
                check_operator(
                    operator=SequentialOperator(
                        steps=[
                            template,
                            SystemFormat(model_input_format=format),
                        ]
                    ),
                    inputs=[instance],
                    targets=[
                        {
                            **instance,
                            "source": required_input,
                            "target": target,
                            "references": [target],
                        }
                    ],
                    tester=self,
                )
