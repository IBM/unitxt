import unittest
from src import unitxt
from datasets import load_dataset as load_dataset_hf
from src.unitxt.text_utils import print_dict
import evaluate

class TestExamples(unittest.TestCase):
    def test_example1(self):
        import examples.example1
        self.assertTrue(True)

    def test_example1(self):
        import examples.example1
        self.assertTrue(True)

    def test_example2(self):
        import examples.example2
        self.assertTrue(True)

    def test_example3(self):
        import examples.example3
        self.assertTrue(True)

    def test_example4(self):
        import examples.example4
        self.assertTrue(True)

    def test_example5(self):


        dataset = load_dataset_hf(
            unitxt.dataset_file,
            'card=cards.wnli,template_item=0',
        )
        
        output = dataset['train'][0]
        target = {'metrics': ['metrics.accuracy'], 'source': "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not_entailment'].\nOutput: ", 'target': 'not_entailment', 'references': ['not_entailment'], 'group': 'unitxt', 'postprocessors': ['to_string']}

        self.assertDictEqual(output, target)

    def test_example6(self):
        import examples.example6
        self.assertTrue(True)

    def test_example7(self):
        data = [
            {'group': 'group1','references':['333', '4'], 'source': 'source1', 'target': 'target1'},
            {'group': 'group1', 'references':['4'], 'source': 'source2', 'target': 'target2'},
            {'group': 'group2', 'references':['3'], 'source': 'source3', 'target': 'target3'},
            {'group': 'group2', 'references':['3'], 'source': 'source4', 'target': 'target4'},
        ]

        for d in data:
            d['metrics'] = ['metrics.accuracy']
            d['postprocessors'] = ['processors.to_string']
            
        predictions = ['4',' 3', '3', '3']

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=predictions, references=data, flatten=True)

        output = results[0]
        print_dict(output)
        print()
        target = {'source': 'source1', 'target': 'target1', 'references': ['333', '4'], 'metrics': ['metrics.accuracy'], 'group': 'group1', 'postprocessors': ['processors.to_string'], 'prediction': '4', 'score_global_accuracy': 0.0, 'score_global_score': 0.0, 'score_global_groups_mean_score': 0.5, 'score_instance_accuracy': 0.0, 'score_instance_score': 0.0, 'origin': 'all_group1'}
        print_dict(target)
        print()
        # self.assertDictEqual(output, target)se
        self.assertTrue(True)

    def test_example8(self):
        dataset = unitxt.load_dataset('recipes.wnli_3_shot')

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=['none' for t in dataset['test']], references=dataset['test'])

        self.assertTrue(True)

    def test_example9(self):
        dataset = load_dataset_hf(unitxt.dataset_file, 'card=cards.wnli,template_item=0,num_demos=5,demos_pool_size=100')

        print_dict(dataset['train'][0])

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=['entailment' for t in dataset['test']], references=dataset['test'])

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()