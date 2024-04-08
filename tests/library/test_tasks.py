from unitxt.task import MultipleChoiceTask
from unitxt.test_utils.operators import apply_operator

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
