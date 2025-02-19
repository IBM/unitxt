from typing import Any, Dict, List

from unitxt.catalog import get_from_catalog
from unitxt.error_utils import UnitxtError
from unitxt.task import Task

from tests.utils import UnitxtTestCase


class TestTasks(UnitxtTestCase):
    def test_task_metrics_type_checking(self):
        operator = Task(
            input_fields={"input": str},
            reference_fields={"label": str},
            prediction_type=str,
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = Dict
        with self.assertRaises(UnitxtError) as e:
            operator.check_metrics_type()
        self.assertIn(
            "The task's prediction type (typing.Dict) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
            str(e.exception),
        )

    def test_single_metric_string_loading(self):
        task = get_from_catalog("tasks.qa.with_context[metrics=metrics.rouge]")
        self.assertListEqual(task.metrics, ["metrics.rouge"])

    def test_multiple_metrics_string_loading(self):
        task = get_from_catalog(
            "tasks.qa.with_context[metrics=[metrics.rouge, metrics.bleu]]"
        )
        self.assertListEqual(task.metrics, ["metrics.rouge", "metrics.bleu"])

    def test_task_metrics_type_checking_with_inputs_outputs(self):
        operator = Task(
            inputs={"input": str},
            outputs={"label": str},
            prediction_type=str,
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = Dict[int, int]
        with self.assertRaises(UnitxtError) as e:
            operator.check_metrics_type()
        self.assertIn(
            "The task's prediction type (typing.Dict[int, int]) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
            str(e.exception),
        )

    def test_task_missing_input_fields(self):
        with self.assertRaises(UnitxtError) as e:
            Task(
                input_fields=None,
                reference_fields={"label": str},
                prediction_type=str,
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertIn(
            "Missing attribute in task: 'input_fields' not set.", str(e.exception)
        )

    def test_task_missing_reference_fields(self):
        with self.assertRaises(UnitxtError) as e:
            Task(
                input_fields={"input": int},
                reference_fields=None,
                prediction_type=str,
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertIn(
            "Missing attribute in task: 'reference_fields' not set.", str(e.exception)
        )

    def test_conflicting_input_fields(self):
        with self.assertRaises(UnitxtError) as e:
            Task(
                inputs={"input": int},
                input_fields={"input": int},
                reference_fields={"label": str},
                prediction_type=str,
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertIn(
            "Conflicting attributes: 'input_fields' cannot be set simultaneously with 'inputs'. Use only 'input_fields'",
            str(e.exception),
        )

    def test_conflicting_output_fields(self):
        with self.assertRaises(UnitxtError) as e:
            Task(
                input_fields={"input": int},
                reference_fields={"label": str},
                outputs={"label": int},
                prediction_type=str,
                metrics=["metrics.wer", "metrics.rouge"],
            )
        self.assertIn(
            "Conflicting attributes: 'reference_fields' cannot be set simultaneously with 'output'. Use only 'reference_fields'",
            str(e.exception),
        )

    def test_set_defaults(self):
        instances = [
            {"input": "Input1", "input_type": "something", "label": 0, "labels": []},
            {"input": "Input2", "label": 1},
        ]

        operator = Task(
            input_fields={"input": str, "input_type": str},
            reference_fields={"label": int, "labels": List[int]},
            prediction_type=Any,
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
            input_fields={"input": str},
            reference_fields={"label": int},
            prediction_type=Any,
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

        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            "The value of 'label' from the 'defaults' must be of "
            "type 'int', however, it is of type 'str'.",
        )

        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            "The value of 'label' from the 'defaults' must be of "
            "type 'int', however, it is of type 'str'.",
        )

    def test_verify_defaults_string_type(self):
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
        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            "The value of 'label' from the 'defaults' must be of "
            "type 'int', however, it is of type 'str'.",
        )
