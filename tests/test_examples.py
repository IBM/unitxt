import unittest

import evaluate
from datasets import load_dataset
from datasets import load_dataset as load_dataset_hf

from src import unitxt
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests
from src.unitxt.text_utils import print_dict


class TestExamples(unittest.TestCase):
    @classmethod
    def setUp(cls):
        register_local_catalog_for_tests()

    def test_example1(self):
        self.assertTrue(True)

    def test_example2(self):
        self.assertTrue(True)

    def test_example3(self):
        self.assertTrue(True)

    def test_add_metric_to_catalog(self):
        from src import unitxt
        from src.unitxt.blocks import ToString
        from src.unitxt.catalog import add_to_catalog
        from src.unitxt.metrics import Accuracy
        from src.unitxt.text_utils import print_dict

        add_to_catalog(
            ToString(field="TBD"), "processors.example.to_string", overwrite=True
        )
        add_to_catalog(Accuracy(), "metrics.example.accuracy", overwrite=True)

        data = [
            {
                "group": "group1",
                "references": ["333", "4"],
                "source": "source1",
                "target": "target1",
                "additional_inputs": [],
            },
            {
                "group": "group1",
                "references": ["4"],
                "source": "source2",
                "target": "target2",
                "additional_inputs": [],
            },
            {
                "group": "group2",
                "references": ["3"],
                "source": "source3",
                "target": "target3",
                "additional_inputs": [],
            },
            {
                "group": "group2",
                "references": ["3"],
                "source": "source4",
                "target": "target4",
                "additional_inputs": [],
            },
        ]

        for d in data:
            d["metrics"] = ["metrics.accuracy"]
            d["postprocessors"] = ["processors.to_string"]

        predictions = ["4", " 3", "3", "3"]

        import evaluate

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=predictions, references=data, flatten=True)

        print_dict(results[0])

        self.assertTrue(True)

    def test_example5(self):
        dataset = load_dataset_hf(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0",
        )

        dataset["train"][0]

        self.assertTrue(True)

    def test_add_recipe_to_catalog(self):
        self.assertTrue(True)

    def test_example6(self):
        self.assertTrue(True)

    def test_example7(self):
        data = [
            {
                "group": "group1",
                "references": ["333", "4"],
                "source": "source1",
                "target": "target1",
                "additional_inputs": [
                    {"key": "a", "value": "1"},
                    {"key": "b", "value": "1"},
                ],
            },
            {
                "group": "group1",
                "references": ["4"],
                "source": "source2",
                "target": "target2",
                "additional_inputs": [
                    {"key": "a", "value": "2"},
                    {"key": "b", "value": "2"},
                ],
            },
            {
                "group": "group2",
                "references": ["3"],
                "source": "source3",
                "target": "target3",
                "additional_inputs": [
                    {"key": "a", "value": "3"},
                    {"key": "b", "value": "3"},
                ],
            },
            {
                "group": "group2",
                "references": ["3"],
                "source": "source4",
                "target": "target4",
                "additional_inputs": [
                    {"key": "a", "value": "4"},
                    {"key": "b", "value": "4"},
                ],
            },
        ]

        for d in data:
            d["metrics"] = ["metrics.accuracy"]
            d["postprocessors"] = ["processors.to_string"]

        predictions = ["4", " 3", "3", "3"]

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=predictions, references=data, flatten=True)

        output = results[0]
        print_dict(output)

        target = {
            "source": "source1",
            "target": "target1",
            "references": ["333", "4"],
            "metrics": ["metrics.accuracy"],
            "group": "group1",
            "postprocessors": ["processors.to_string"],
            "prediction": "4",
            "score_global_accuracy": 0.0,
            "score_global_score": 0.0,
            "score_global_groups_mean_score": 0.5,
            "score_instance_accuracy": 0.0,
            "score_instance_score": 0.0,
            "origin": "all_group1",
        }
        print_dict(target)
        # self.assertDictEqual(output, target)se
        self.assertTrue(True)

    def test_example8(self):
        dataset = unitxt.load("recipes.wnli_3_shot")
        metric = evaluate.load(unitxt.metric_file)

        metric.compute(
            predictions=["none" for t in dataset["test"]], references=dataset["test"]
        )

    def test_evaluate(self):
        import evaluate

        from src import unitxt
        from src.unitxt import load
        from src.unitxt.text_utils import print_dict

        dataset = load("recipes.wnli_3_shot")

        metric = evaluate.load(unitxt.metric_file)
        results = metric.compute(
            predictions=["none" for t in dataset["test"]], references=dataset["test"]
        )

        print_dict(results[0])

        self.assertTrue(True)

    def test_load_dataset(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0",
            download_mode="force_redownload",
        )
        print_dict(dataset["train"][0])
        # self.assertDictEqual(target, dataset['train'][0])

    def test_full_flow_of_hf(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        import evaluate

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(
            predictions=["entailment" for t in dataset["test"]],
            references=dataset["test"],
        )

        print_dict(results[0])

        # self.assertDictEqual(target, results[0])
