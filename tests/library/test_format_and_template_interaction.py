from src.unitxt.formats import SystemFormat
from src.unitxt.operators import SequentialOperator
from src.unitxt.templates import InputOutputTemplate
from src.unitxt.test_utils.operators import check_operator
from tests.utils import UnitxtTestCase

instance = {"inputs": {"question": "what?"}, "outputs": {"answer": "that!"}}


class TestFormatAndTemplateInteraction(UnitxtTestCase):
    def test_simple_interaction(self):
        operator = SequentialOperator(
            steps=[
                InputOutputTemplate(
                    input_format="Question: {question}",
                    output_format="Answer: {answer}",
                ),
                SystemFormat(model_input_format="<user>\n{source}\n<agent>\n"),
            ]
        )

        source = "<user>\nQuestion: what?\n<agent>\n"
        target = "Answer: that!"

        check_operator(
            operator=operator,
            inputs=[instance],
            targets=[
                {**instance, "source": source, "target": target, "references": [target]}
            ],
            tester=self,
        )

    def test_new_line_in_output_interaction(self):
        operator = SequentialOperator(
            steps=[
                InputOutputTemplate(
                    input_format="Question:\n{question}",
                    output_format="Answer:\n{answer}",
                ),
                SystemFormat(model_input_format="<user>\n{source}\n<agent>\n"),
            ]
        )

        source = "<user>\nQuestion:\nwhat?\n<agent>\n"
        target = "Answer:\nthat!"

        check_operator(
            operator=operator,
            inputs=[instance],
            targets=[
                {**instance, "source": source, "target": target, "references": [target]}
            ],
            tester=self,
        )

    def test_new_line_in_input_interaction(self):
        operator = SequentialOperator(
            steps=[
                InputOutputTemplate(
                    input_format="Question:\n{question}\nAnswer:\n",
                    output_format="{answer}",
                ),
                SystemFormat(model_input_format="<user>\n{source}\n<agent>\n"),
            ]
        )

        source = "<user>\nQuestion:\nwhat?\nAnswer:\n\n<agent>\n"
        target = "that!"

        check_operator(
            operator=operator,
            inputs=[instance],
            targets=[
                {**instance, "source": source, "target": target, "references": [target]}
            ],
            tester=self,
        )

    def test_new_line_in_input_interaction_capital_n(self):
        operator = SequentialOperator(
            steps=[
                InputOutputTemplate(
                    input_format="Question:\n{question}\nAnswer:\n",
                    output_format="{answer}",
                ),
                SystemFormat(model_input_format="<user>\n{source}\\N<agent>\n"),
            ]
        )

        source = "<user>\nQuestion:\nwhat?\nAnswer:\n<agent>\n"
        target = "that!"

        check_operator(
            operator=operator,
            inputs=[instance],
            targets=[
                {**instance, "source": source, "target": target, "references": [target]}
            ],
            tester=self,
        )
