from unitxt.task import Task

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
