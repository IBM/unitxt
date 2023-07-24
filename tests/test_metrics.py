import unittest

from src.unitxt.test_utils.metrics import apply_metric
from src.unitxt.metrics import Accuracy, Squad, F1, F1Micro, F1Macro


class TestMetrics(unittest.TestCase):

    def test_accuracy(self):
       
        metric = Accuracy()
        
        predictions = ['A', 'B', 'C']
        references = [['B'], ['A'], ['C']]
        
        instance_targets = [{'accuracy': 0.0, 'score': 0.0},
                            {'accuracy': 0.0, 'score': 0.0},
                            {'accuracy': 1.0, 'score': 1.0}]
        
        global_target = {'accuracy': 1/3, 'score': 1/3}
        
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertDictEqual(outputs[0]['score']['global'], global_target)
        for output, target in zip(outputs, instance_targets):
            self.assertDictEqual(output['score']['instance'], target)

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


    def test_f1(self):
        # metric = F1()
        # predictions = ['1976', 'Beyonce', 'climate change']
        # references = [['1976'], ['Beyoncé and Bruno Mars'], ['climate change']]
        # instance_targets = [{'f1': 1.0,'score': 1.0},
        #                     {'f1': 0.0,'score': 0.0},
        #                     {'f1': 1.0,'score': 1.0}]
        # global_target =  2 /3
        # outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        # self.assertAlmostEqual(global_target, outputs[0]['score']['global']['score'])
        # for output, target in zip(outputs, instance_targets):
        #     self.assertEqual(output['score']['instance'], target)

        metric = F1()
        references = [['0'], ['1'], ['0'], ['1'], ['0']]
        predictions = ['0', '0', '1', '1', '0']
        global_target = 0.5
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertAlmostEqual(global_target, outputs[0]['score']['global']['score'])


    def test_f1_micro(self):
        metric = F1Micro()
        predictions = ['1976', 'Beyonce', 'climate change']
        references = [['1976'], ['Beyoncé and Bruno Mars'], ['climate change']]
        instance_targets = [{'f1': 1.0,'score': 1.0},
                            {'f1': 0.0,'score': 0.0},
                            {'f1': 1.0,'score': 1.0}]
        global_target =  2 /3
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

    def test_f1_macro(self):
        metric = F1Macro()
        predictions = ['0', '2', '1', '0', '0', '1']
        references = [['0'], ['1'], ['2'], ['0'], ['1'], ['2']]
        global_target = 0.26666666666
        outputs = apply_metric(metric=metric, predictions=predictions, references=references)

        self.assertAlmostEqual(global_target, outputs[0]['score']['global']['score'])
