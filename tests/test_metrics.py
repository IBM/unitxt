import unittest

from src.unitxt.test_utils.metrics import apply_metric
from src.unitxt.metrics import Accuracy, Squad


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

    def test_squad(self):
        metric = Squad()
        predictions = ['1976', 'Beyonce', 'climate change']
        references = [['1976'], ['Beyoncé and Bruno Mars'], ['climate change']]
        instance_targets = [{'exact_match': 100.0, 'f1': 100.0,'score': 100.0},
                            {'exact_match': 0.0, 'f1': 0.0,'score': 0.0},
                            {'exact_match': 100.0, 'f1': 100.0,'score': 100.0}]
        global_target = 100 * 2 /3
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertAlmostEqual(global_target, outputs[0]['score']['global']['score'])
        for output, target in zip(outputs, instance_targets):
            self.assertEqual(output['score']['instance'], target)

