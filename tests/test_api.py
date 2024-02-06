import unittest

from src.unitxt.api import evaluate, load_dataset


class TestAPI(unittest.TestCase):
    def test_load_dataset(self):
        dataset = load_dataset(
            "card=cards.stsb,template=templates.regression.two_texts.simple,max_train_instances=5,max_validation_instances=5,max_test_instances=5"
        )
        instance = {
            "metrics": ["metrics.spearman"],
            "source": "Given this sentence: 'A plane is taking off.', on a scale of 1 to 5, what is the similarity to this text An air plane is taking off.?\n",
            "target": "5.0",
            "references": ["5.0"],
            "additional_inputs": {
                "key": [
                    "text1",
                    "text2",
                    "attribute_name",
                    "min_value",
                    "max_value",
                    "attribute_value",
                ],
                "value": [
                    "A plane is taking off.",
                    "An air plane is taking off.",
                    "similarity",
                    "1",
                    "5",
                    "5.0",
                ],
            },
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
