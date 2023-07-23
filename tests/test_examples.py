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
                  'source': "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not_entailment'].\nOutput: ",
                  'target': 'not_entailment', 'references': ['not_entailment'], 'group': 'unitxt',
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
                      'source': 'Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is 0, 1.\nOutput: ',
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
                'source': "Input: Given this sentence: I couldn't put the pot on the shelf because it was too tall., classify if this sentence: The pot was too tall. is 0, 1.\nOutput: not entailment\n\nInput: Given this sentence: No one joins Facebook to be sad and lonely. But a new study from the University of Wisconsin psychologist George Lincoln argues that that's exactly how it makes us feel., classify if this sentence: That's exactly how the study makes us feel. is 0, 1.\nOutput: entailment\n\nInput: Given this sentence: The large ball crashed right through the table because it was made of steel., classify if this sentence: The large ball was made of steel. is 0, 1.\nOutput: not entailment\n\nInput: Given this sentence: John couldn't see the stage with Billy in front of him because he is so short., classify if this sentence: John is so short. is 0, 1.\nOutput: not entailment\n\nInput: Given this sentence: The man couldn't lift his son because he was so weak., classify if this sentence: The man was so weak. is 0, 1.\nOutput: not entailment\n\nInput: Given this sentence: The drain is clogged with hair. It has to be cleaned., classify if this sentence: The hair has to be cleaned. is 0, 1.\nOutput: ",
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
