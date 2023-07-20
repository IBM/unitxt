import unittest

from src.unitxt.test_utils.metrics import apply_metric
from src.unitxt.metrics import Accuracy


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
        metric = Accuracy()
        predictions = ['A', 'B', 'C']
        references = [['B'], ['A'], ['C']]
        instance_targets = [{'accuracy': 0.0,'score': 0.0},
                            {'accuracy': 0.0,'score': 0.0},
                            {'accuracy': 1.0,'score': 1.0}]
        global_target = 1/3
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertEqual(global_target, outputs[0]['score']['global']['score'])
        for output, target in zip(outputs, instance_targets):
            self.assertEqual(output['score']['instance'], target)

