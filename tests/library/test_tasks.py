from typing import Any, Dict, List

from unitxt.task import Task

from tests.utils import UnitxtTestCase


class TestTasks(UnitxtTestCase):
    def test_task_metrics_type_checking(self):
        operator = Task(
            inputs={"input": str},
            outputs={"label": str},
            prediction_type=str,
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = Dict[int, int]
        with self.assertRaises(ValueError) as e:
            operator.check_metrics_type()
        self.assertEqual(
            str(e.exception),
            "The task's prediction type (typing.Dict[int, int]) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
        )

    def test_set_defaults(self):
        instances = [
            {"input": "Input1", "input_type": "something", "label": 0, "labels": []},
            {"input": "Input2", "label": 1},
        ]

        operator = Task(
            inputs={"input": str, "input_type": str},
            outputs={"label": int, "labels": List[int]},
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
            inputs={"input": str},
            outputs={"label": int},
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
            f"key in either 'inputs' or 'outputs'. However, the name '{default_name}' "
            f"was provided which does not match any of the keys.",
        )

        default_name = "label"
        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            f"The value of '{default_name}' from the 'defaults' must be of "
            f"type 'int', however, it is of type '{type(operator.defaults[default_name])}'.",
        )

    def test_verify_defaults_string_type(self):
        operator = Task(
            inputs={"input": "str"},
            outputs={"label": "int"},
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
            f"key in either 'inputs' or 'outputs'. However, the name '{default_name}' "
            f"was provided which does not match any of the keys.",
        )

        default_name = "label"
        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            f"The value of '{default_name}' from the 'defaults' must be of "
            f"type 'int', however, it is of type '{type(operator.defaults[default_name])}'.",
        )
