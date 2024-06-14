from unitxt.task import SubTask, Task

from tests.utils import UnitxtTestCase


class TestTasks(UnitxtTestCase):
    def test_task_metrics_type_checking(self):
        operator = Task(
            inputs={"input": "str"},
            outputs={"label": "str"},
            prediction_type="str",
            metrics=["metrics.wer", "metrics.rouge"],
        )

        operator.check_metrics_type()

        operator.prediction_type = "Dict"
        with self.assertRaises(ValueError) as e:
            operator.check_metrics_type()
        self.assertEqual(
            str(e.exception),
            "The task's prediction type (typing.Dict) and 'metrics.wer' metric's prediction type "
            "(<class 'str'>) are different.",
        )

    def test_set_defaults(self):
        instances = [
            {"input": "Input1", "input_type": "something", "label": 0, "labels": []},
            {"input": "Input2", "label": 1, "labels": [0, 0, 0]},
        ]

        operator = Task(
            inputs={"input": "str", "input_type": "str"},
            outputs={"label": "int", "labels": "List[int]"},
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
                    "input_type": "text",
                    "label": 0,
                    "labels": [0, 1, 2],
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
        val_type = "int"
        operator.defaults = {"label": "LABEL"}
        with self.assertRaises(AssertionError) as e:
            operator.verify_defaults()
        self.assertEqual(
            str(e.exception),
            f"The value of '{default_name}' from the 'defaults' must be of "
            f"type '{val_type}', however, it is of type '{type(operator.defaults[default_name])}'.",
        )

    def test_sub_task_creation(self):
        sub_task = Task(
            inputs={"code": "int", "type_of_output": "List[str]"},
            outputs={"processed_code": "int"},
            prediction_type="str",
            metrics=["metrics.accuracy"],
            defaults={"code": 1, "processed_code": -1},
        )
        main_task = SubTask(
            sub_tasks=["tasks.generation", "tasks.translation.directed", sub_task],
            inputs={"type_of_input": "List[str]"},
            outputs={"time": "float", "quantity": "int"},
            defaults={
                "processed_code": 0,
                "source_language": "en",
                "target_language": "en",
                "time": 0.5,
            },
        )

        instance = {
            "type_of_output": [""],
            "text": "Some Text",
            "input": "Some Input",
            "translation": "Some Translation",
            "output": "Some Output",
            "quantity": 10,
            "type_of_input": [""],
            "code": -3,
        }

        processed_instance = main_task.process(instance)
        processed_metrics = processed_instance.pop("metrics")
        metrics = ["metrics.accuracy", "metrics.normalized_sacrebleu"]
        self.assertEqual(
            processed_instance,
            {
                "inputs": {
                    "input": "Some Input",
                    "type_of_input": [""],
                    "type_of_output": [""],
                    "text": "Some Text",
                    "source_language": "en",
                    "target_language": "en",
                    "code": 1,
                },
                "outputs": {
                    "output": "Some Output",
                    "translation": "Some Translation",
                    "processed_code": 0,
                    "time": 0.5,
                    "quantity": 10,
                },
                "data_classification_policy": [],
            },
        )
        processed_metrics.sort()
        metrics.sort()
        self.assertListEqual(processed_metrics, metrics)

    def test_misconfigured_sub_task(self):
        prediction_type = "int"
        prediction_types = ["str"]
        with self.assertRaises(AssertionError) as e:
            SubTask(sub_tasks=["tasks.generation"], prediction_type=prediction_type)
        self.assertEqual(
            str(e.exception),
            f"The specified 'prediction_type' is '{prediction_type}' and it needs "
            f"to be consistent with prediction types of given sub-tasks. However, "
            f"sub tasks have the following prediction types: '{prediction_types}'.",
        )
