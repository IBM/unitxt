from typing import List

from src.unitxt.task import FormTask, MultipleChoiceTask
from src.unitxt.test_utils.operators import apply_operator
from tests.utils import UnitxtTestCase


class TestTasks(UnitxtTestCase):
    def test_multiple_choice_task(self):
        operator = MultipleChoiceTask(
            inputs=["choices", "sentence1", "sentence2"],
            outputs=["label"],
            metrics=["metrics.accuracy"],
        )

        inputs = [
            {
                "choices": ["entailment", "not_entailment"],
                "sentence1": "This is a sentence",
                "sentence2": "This is another sentence",
                "label": "entailment",
            },
            {
                "choices": ["entailment", "not_entailment"],
                "sentence1": "This is a sentence",
                "sentence2": "This is another sentence",
                "label": "not_entailment",
            },
        ]

        targets = [
            {
                "inputs": {
                    "choices": "A. entailment\nB. not_entailment",
                    "sentence1": "This is a sentence",
                    "sentence2": "This is another sentence",
                },
                "outputs": {"label": "A"},
                "metrics": ["metrics.accuracy"],
            },
            {
                "inputs": {
                    "choices": "A. entailment\nB. not_entailment",
                    "sentence1": "This is a sentence",
                    "sentence2": "This is another sentence",
                },
                "outputs": {"label": "B"},
                "metrics": ["metrics.accuracy"],
            },
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_task_metrics_type_checking(self):
        operator = FormTask(
            inputs={"input": str},
            outputs={"label": str},
            prediction_type="str",
            metrics=["metrics.wer", "metrics.rouge", "metrics.roc_auc"],
        )

        operator.check_metrics_type()

        operator.prediction_type = "Dict"
        with self.assertRaises(ValueError):
            operator.check_metrics_type()

    def test_task_instance_value_type_checking(self):
        operator = FormTask(
            inputs={"input1": List[str], "input2": "Tuple[str]"},
            outputs={"label": int},
            prediction_type="Any",
            metrics=["metrics.accuracy"],
        )

        inputs = [
            {
                "input1": ["Test", "testing"],
                "input2": ("tasks", "task"),
                "label": 1,
            },
        ]
        targets = [
            {
                "inputs": {
                    "input1": ["Test", "testing"],
                    "input2": ("tasks", "task"),
                },
                "outputs": {"label": 1},
                "metrics": ["metrics.accuracy"],
            },
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)
        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

        inputs[0].update({"input1": "Test"})
        with self.assertRaises(ValueError):
            apply_operator(operator, inputs)
