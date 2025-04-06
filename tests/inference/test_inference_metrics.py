from unitxt.logging_utils import get_logger
from unitxt.metrics import (
    BertScore,
)
from unitxt.settings_utils import get_settings
from unitxt.test_utils.metrics import test_metric

from tests.utils import UnitxtInferenceTestCase

logger = get_logger()
settings = get_settings()


class TestInferenceMetrics(UnitxtInferenceTestCase):
    def test_bert_score_deberta_base_mnli(self):
        metric = BertScore(model_name="microsoft/deberta-base-mnli")
        predictions = ["hello there general dude", "foo bar foobar"]
        references = [
            ["hello there general kenobi", "hello there!"],
            ["foo bar foobar", "foo bar"],
        ]
        instance_targets = [
            {
                "f1": 0.81,
                "precision": 0.85,
                "recall": 0.81,
                "score": 0.81,
                "score_name": "f1",
            },
            {
                "f1": 1.0,
                "precision": 1.0,
                "recall": 1.0,
                "score": 1.0,
                "score_name": "f1",
            },
        ]
        global_target = {
            "f1": 0.9,
            "f1_ci_high": 1.0,
            "f1_ci_low": 0.81,
            "precision": 0.93,
            "precision_ci_high": 1.0,
            "precision_ci_low": 0.85,
            "recall": 0.91,
            "recall_ci_high": 1.0,
            "recall_ci_low": 0.81,
            "score": 0.9,
            "score_ci_high": 1.0,
            "score_ci_low": 0.81,
            "score_name": "f1",
            "num_of_instances": 2,
        }
        test_metric(
            metric=metric,
            predictions=predictions,
            references=references,
            instance_targets=instance_targets,
            global_target=global_target,
        )
