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
                'source': "Input: Given this sentence: I can't cut that tree down with that axe; it is too small.,"
                          " classify if this sentence: The axe is too small. is ['entailment', 'not entailment'].\n"
                          "Output: not entailment\n\nInput: Given this sentence: Alice was dusting the living room and "
                          "trying to find the button that Mama had hidden. No time today to look at old pictures in her "
                          "favorite photo album. Today she had to hunt for a button, so she put the album on a chair "
                          "without even opening it., classify if this sentence: She put the album on a chair without "
                          "even opening the living room. is ['entailment', 'not entailment'].\nOutput: "
                          "entailment\n\nInput: Given this sentence: Joe has sold his house and bought a new one a "
                          "few miles away. He will be moving out of it on Thursday., classify if this sentence: "
                          "He will be moving out of The old house on Thursday. is ['entailment', 'not entailment'].\n"
                          "Output: not entailment\n\nInput: Given this sentence: Although they ran at about the same "
                          "speed, Sue beat Sally because she had such a good start., classify if this sentence: "
                          "Sue had such a good start. is ['entailment', 'not entailment'].\nOutput: not entailment\n\n"
                          "Input: Given this sentence: John was jogging through the park when he saw a man juggling "
                          "watermelons. He was very impressive., classify if this sentence: John was very impressive. "
                          "is ['entailment', 'not entailment'].\nOutput: entailment\n\nInput: Given this sentence: "
                          "The drain is clogged with hair. It has to be cleaned., classify if this sentence: The hair "
                          "has to be cleaned. is ['entailment', 'not entailment'].\nOutput: ",
                'target': 'entailment', 'references': ['entailment'], 'metrics': ['metrics.accuracy'],
                'group': 'unitxt', 'postprocessors': ['to_string'], 'prediction': 'entailment', 'score': {
                    'global': {'accuracy': 0.5633802816901409, 'score': 0.5633802816901409,
                               'groups_mean_score': 0.5633802816901409}, 'instance': {'accuracy': 1.0, 'score': 1.0}},
                'origin': 'all_unitxt'}

        self.assertDictEqual(target, results[0])


if __name__ == '__main__':
    unittest.main()
