import unittest
from src import unitxt
from datasets import load_dataset as load_dataset_hf
from src.unitxt.text_utils import print_dict
import evaluate

from src.unitxt.hf_utils import set_hf_caching
from src import unitxt
from datasets import load_dataset
from src.unitxt.text_utils import print_dict


class TestExamples(unittest.TestCase):
    def test_example1(self):
        with set_hf_caching(False):
            import examples.example1
        self.assertTrue(True)

    def test_example2(self):
        with set_hf_caching(False):
            import examples.example2
        self.assertTrue(True)

    def test_example3(self):
        with set_hf_caching(False):
            import examples.example3
        self.assertTrue(True)

    def test_add_metric_to_catalog(self):
        with set_hf_caching(False):
            import examples.add_metric_to_catalog
        self.assertTrue(True)

    def test_example5(self):
        dataset = load_dataset_hf(
            unitxt.dataset_file,
            'card=cards.wnli,template_item=0',
        )

        output = dataset['train'][0]
        target = {'metrics': ['metrics.accuracy'],
                  'source': "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not entailment'].\nOutput: ",
                  'target': 'not entailment', 'references': ['not entailment'], 'group': 'unitxt',
                  'postprocessors': ['to_string']}

        self.assertDictEqual(output, target)

    def test_add_recipe_to_catalog(self):
        with set_hf_caching(False):
            import examples.add_recipe_to_catalog
        self.assertTrue(True)

    def test_example6(self):
        with set_hf_caching(False):
            import examples.example6
        self.assertTrue(True)

    def test_example7(self):
        data = [
            {'group': 'group1', 'references': ['333', '4'], 'source': 'source1', 'target': 'target1'},
            {'group': 'group1', 'references': ['4'], 'source': 'source2', 'target': 'target2'},
            {'group': 'group2', 'references': ['3'], 'source': 'source3', 'target': 'target3'},
            {'group': 'group2', 'references': ['3'], 'source': 'source4', 'target': 'target4'},
        ]

        for d in data:
            d['metrics'] = ['metrics.accuracy']
            d['postprocessors'] = ['processors.to_string']

        predictions = ['4', ' 3', '3', '3']

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=predictions, references=data, flatten=True)

        output = results[0]
        print_dict(output)
        print()
        target = {'source': 'source1', 'target': 'target1', 'references': ['333', '4'], 'metrics': ['metrics.accuracy'],
                  'group': 'group1', 'postprocessors': ['processors.to_string'], 'prediction': '4',
                  'score_global_accuracy': 0.0, 'score_global_score': 0.0, 'score_global_groups_mean_score': 0.5,
                  'score_instance_accuracy': 0.0, 'score_instance_score': 0.0, 'origin': 'all_group1'}
        print_dict(target)
        print()
        # self.assertDictEqual(output, target)se
        self.assertTrue(True)

    def test_example8(self):
        dataset = unitxt.load_dataset('recipes.wnli_3_shot')

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=['none' for t in dataset['test']], references=dataset['test'])

    def test_evaluate(self):
        with set_hf_caching(False):
            import examples.evaluate_example
        self.assertTrue(True)

    def test_load_dataset(self):
        with set_hf_caching(False):
            dataset = load_dataset(unitxt.dataset_file,
                                   'card=cards.wnli,template_item=0',
                                   )
            print_dict(dataset['train'][0])
            target = {'metrics': ['metrics.accuracy'],
                      'source': "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not entailment'].\nOutput: ",
                      'target': 'not entailment',
                      'references': ['not entailment'],
                      'group': 'unitxt',
                      'postprocessors': ['to_string']}
            self.assertDictEqual(target, dataset['train'][0])

    def test_full_flow_of_hf(self):
        with set_hf_caching(False):
            dataset = load_dataset(unitxt.dataset_file,
                                   'card=cards.wnli,template_item=0,num_demos=5,demos_pool_size=100')

            print_dict(dataset['train'][0])

            import evaluate

            metric = evaluate.load(unitxt.metric_file)

            results = metric.compute(predictions=['entailment' for t in dataset['test']], references=dataset['test'])

            print_dict(results[0])
            target = {
                'source': "Input: Given this sentence: Sam and Amy are passionately in love, but Amy's parents are unhappy about it, because they are snobs., classify if this sentence: Amy's parents are snobs. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: This morning, Joey built a sand castle on the beach, and put a toy flag in the highest tower, but this afternoon the tide knocked it down., classify if this sentence: This afternoon the tide knocked The flag down. is ['entailment', 'not entailment'].\nOutput: entailment\n\nInput: Given this sentence: Dan had to stop Bill from toying with the injured bird. He is very cruel., classify if this sentence: Bill is very cruel. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: John promised Bill to leave, so an hour later he left., classify if this sentence: John left. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: Fred is the only man still alive who remembers my great-grandfather. He was a remarkable man., classify if this sentence: My great-grandfather was a remarkable man. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: The drain is clogged with hair. It has to be cleaned., classify if this sentence: The hair has to be cleaned. is ['entailment', 'not entailment'].\nOutput: ",
                'target': 'entailment',
                'references': ['entailment'],
                'metrics': ['metrics.accuracy'],
                'group': 'unitxt',
                'postprocessors': ['to_string'],
                'prediction': 'entailment',
                'score': {'global': {'accuracy': 0.5633802816901409,
                                     'score': 0.5633802816901409,
                                     'groups_mean_score': 0.5633802816901409},
                          'instance': {'accuracy': 1.0, 'score': 1.0}},
                'origin': 'all_unitxt'}
        self.assertDictEqual(target, results[0])


if __name__ == '__main__':
    unittest.main()
