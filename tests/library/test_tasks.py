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

    def test_sub_task_creation(self):
        sub_task = Task(
            inputs={"type_of_output": "List[str]"},
            outputs={"code": "int"},
            prediction_type="str",
            metrics=["metrics.accuracy"],
        )
        main_task = SubTask(
            sub_tasks=["tasks.generation", "tasks.translation.directed", sub_task],
            inputs={"type_of_input": "List[str]"},
            outputs={"quantity": "int"},
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
            "source_language": "en",
            "target_language": "en",
        }

        processed_instance = main_task.process(instance)
        processed_metrics = processed_instance.pop("metrics")
        metrics = ["metrics.accuracy", "metrics.normalized_sacrebleu"]
        self.assertEqual(
            processed_instance,
            {
                "inputs": {
                    "input": "Some Input",
                    "source_language": "en",
                    "target_language": "en",
                    "text": "Some Text",
                    "type_of_input": [""],
                    "type_of_output": [""],
                },
                "outputs": {
                    "code": -3,
                    "output": "Some Output",
                    "quantity": 10,
                    "translation": "Some Translation",
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
