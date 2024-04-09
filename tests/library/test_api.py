from unitxt.api import evaluate, load_dataset

from tests.utils import UnitxtTestCase


class TestAPI(UnitxtTestCase):
    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1 to 5, what is the similarity to this text An air plane is taking off.?\n",
            "target": "5.0",
            "references": ["5.0"],
            "task_data": '{"text1": "A plane is taking off.", "text2": "An air plane is taking off.", "attribute_name": "similarity", "min_value": "1", "max_value": "5", "attribute_value": 5.0}',
            "group": "unitxt",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertEqual(len(dataset["train"]), 5)
        self.assertDictEqual(dataset["train"][0], instance)

    def test_evaluate(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertAlmostEqual(results[0]["score"]["global"]["score"], 0.026, 3)

    def test_evaluate_with_metrics_external_setup(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5,metrics=[metrics.accuracy],postprocessors=[processors.first_character]"
        )
        self.assertEqual(dataset["train"][0]["metrics"], ["metrics.accuracy"])
        self.assertEqual(
            dataset["train"][0]["postprocessors"], ["processors.first_character"]
        )
        predictions = ["2.5", "2.5", "2.2", "3", "4"]
        results = evaluate(predictions, dataset["train"])
        self.assertAlmostEqual(results[0]["score"]["global"]["score"], 0.2, 3)
