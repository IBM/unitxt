from unitxt.error_utils import UnitxtError
from unitxt.task import Task

from tests.utils import UnitxtTestCase


class TestTasks(UnitxtTestCase):
    def test_task_metrics_type_checking(self):
        operator = Task(
            input_fields={"input": "str"},
            reference_fields={"label": "str"},
            prediction_type="str",
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = "Dict"
        with self.assertRaises(UnitxtError) as e:
            operator.check_metrics_type()
        self.assertIn(
            "The task's prediction type (typing.Dict) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
            str(e.exception),
        )

    def test_task_metrics_type_checking_with_inputs_outputs(self):
        operator = Task(
            inputs={"input": "str"},
            outputs={"label": "str"},
            prediction_type="str",
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = "Dict"
        with self.assertRaises(UnitxtError) as e:
            operator.check_metrics_type()
        self.assertIn(
            "The task's prediction type (typing.Dict) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
            str(e.exception),
        )

    def test_task_missing_input_fields(self):
        with self.assertRaises(ValueError) as e:
            Task(
                input_fields=None,
                reference_fields={"label": "str"},
                prediction_type="str",
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertEqual(
            str(e.exception), "Missing attribute in task: 'input_fields' not set."
        )

    def test_task_missing_reference_fields(self):
        with self.assertRaises(ValueError) as e:
            Task(
                input_fields={"input": "int"},
                reference_fields=None,
                prediction_type="str",
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertIn(
            str(e.exception), "Missing attribute in task: 'reference_fields' not set."
        )

    def test_conflicting_input_fields(self):
        with self.assertRaises(ValueError) as e:
            Task(
                inputs={"input": "int"},
                input_fields={"input": "int"},
                reference_fields={"label": "str"},
                prediction_type="str",
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertEqual(
            str(e.exception),
            "Conflicting attributes: 'input_fields' cannot be set simultaneously with 'inputs'. Use only 'input_fields'",
        )

    def test_conflicting_output_fields(self):
        with self.assertRaises(ValueError) as e:
            Task(
                input_fields={"input": "int"},
                reference_fields={"label": "str"},
                outputs={"label": "int"},
                prediction_type="str",
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertEqual(
            str(e.exception),
            "Conflicting attributes: 'reference_fields' cannot be set simultaneously with 'output'. Use only 'reference_fields'",
        )

    def test_set_defaults(self):
        instances = [
            {"input": "Input1", "input_type": "something", "label": 0, "labels": []},
            {"input": "Input2", "label": 1},
        ]

        operator = Task(
            input_fields={"input": "str", "input_type": "str"},
            reference_fields={"label": "int", "labels": "List[int]"},
            prediction_type="Any",
            metrics=["metrics.accuracy"],
            defaults={"input_type": "text", "labels": [0, 1, 2]},
        )

        processed_instances = [
            operator.set_default_values(instance) for instance in instances
        ]
        self.assertEqual(
            processed_instances,
            [
                {
                    "input": "Input1",
                    "input_type": "something",
                    "label": 0,
                    "labels": [],
                },
                {
                    "input": "Input2",
                    "label": 1,
                    "labels": [0, 1, 2],
                    "input_type": "text",
                },
            ],
        )

    def test_verify_defaults(self):
        operator = Task(
            input_fields={"input": "str"},
            reference_fields={"label": "int"},
            prediction_type="Any",
            metrics=["metrics.accuracy"],
        )

        default_name = "input_type"
        operator.defaults = {"input_type": "text"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            f"If specified, all keys of the 'defaults' must refer to a chosen "
            f"key in either 'input_fields' or 'reference_fields'. However, the name '{default_name}' "
            f"was provided which does not match any of the keys.",
        )

        default_name = "label"
        val_type = "int"
        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            f"The value of '{default_name}' from the 'defaults' must be of "
            f"type '{val_type}', however, it is of type '{type(operator.defaults[default_name])}'.",
        )
