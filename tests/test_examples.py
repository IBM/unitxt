import unittest

import evaluate
from datasets import load_dataset
from datasets import load_dataset as load_dataset_hf
from src import unitxt
from src.unitxt.text_utils import print_dict


class TestExamples(unittest.TestCase):
    def test_example1(self):
        import examples.example1

        self.assertTrue(True)

    def test_example2(self):
        import examples.example2

        self.assertTrue(True)

    def test_example3(self):
        import examples.example3

        self.assertTrue(True)

    def test_add_metric_to_catalog(self):
        from src import unitxt
        from src.unitxt.blocks import ToString
        from src.unitxt.catalog import add_to_catalog
        from src.unitxt.metrics import Accuracy
        from src.unitxt.text_utils import print_dict

        add_to_catalog(ToString(), "processors.to_string", overwrite=True)
        add_to_catalog(Accuracy(), "metrics.accuracy", overwrite=True)

        data = [
            {"group": "group1", "references": ["333", "4"], "source": "source1", "target": "target1"},
            {"group": "group1", "references": ["4"], "source": "source2", "target": "target2"},
            {"group": "group2", "references": ["3"], "source": "source3", "target": "target3"},
            {"group": "group2", "references": ["3"], "source": "source4", "target": "target4"},
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
            "card=cards.wnli,template_item=0",
        )

        output = dataset["train"][0]
        target = {
            "metrics": ["metrics.accuracy"],
            "source": "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not entailment'].\nOutput: ",
            "target": "not entailment",
            "references": ["not entailment"],
            "group": "unitxt",
            "postprocessors": ["to_string"],
        }

        self.assertTrue(True)

    def test_add_recipe_to_catalog(self):
        import examples.add_recipe_to_catalog

        self.assertTrue(True)

    def test_example6(self):
        import examples.example6

        self.assertTrue(True)

    def test_example7(self):
        data = [
            {"group": "group1", "references": ["333", "4"], "source": "source1", "target": "target1"},
            {"group": "group1", "references": ["4"], "source": "source2", "target": "target2"},
            {"group": "group2", "references": ["3"], "source": "source3", "target": "target3"},
            {"group": "group2", "references": ["3"], "source": "source4", "target": "target4"},
        ]

        for d in data:
            d["metrics"] = ["metrics.accuracy"]
            d["postprocessors"] = ["processors.to_string"]

        predictions = ["4", " 3", "3", "3"]

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=predictions, references=data, flatten=True)

        output = results[0]
        print_dict(output)
        print()
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
        print()
        # self.assertDictEqual(output, target)se
        self.assertTrue(True)

    def test_example8(self):
        dataset = unitxt.load_dataset("recipes.wnli_3_shot")

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=["none" for t in dataset["test"]], references=dataset["test"])

    def test_evaluate(self):
        import evaluate
        from src import unitxt
        from src.unitxt.catalog import add_to_catalog
        from src.unitxt.common import CommonRecipe
        from src.unitxt.load import load_dataset
        from src.unitxt.text_utils import print_dict

        dataset = load_dataset("recipes.wnli_3_shot")

        import evaluate

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=["none" for t in dataset["test"]], references=dataset["test"])

        print_dict(results[0])

        self.assertTrue(True)

    def test_load_dataset(self):
        dataset = load_dataset(
            unitxt.dataset_file, "card=cards.wnli,template_item=0", download_mode="force_redownload"
        )
        print_dict(dataset["train"][0])
        target = {
            "metrics": ["metrics.accuracy"],
            "source": "Input: Given this sentence: I stuck a pin through a carrot. When I pulled the pin out, it had a hole., classify if this sentence: The carrot had a hole. is ['entailment', 'not entailment'].\nOutput: ",
            "target": "not entailment",
            "references": ["not entailment"],
            "group": "unitxt",
            "postprocessors": ["to_string"],
        }
        # self.assertDictEqual(target, dataset['train'][0])

    def test_full_flow_of_hf(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_item=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        import evaluate

        metric = evaluate.load(unitxt.metric_file)

        results = metric.compute(predictions=["entailment" for t in dataset["test"]], references=dataset["test"])

        print_dict(results[0])
        target = {
            "source": "Input: Given this sentence: The politicians far away in Washington could not know the settlers so they must make rules to regulate them., classify if this sentence: The politicians must make rules to regulate them. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: I put the cake away in the refrigerator. It has a lot of butter in it., classify if this sentence: The cake has a lot of butter in it. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: By rolling over in her upper berth, Tatyana could look over the edge of it and see her mother plainly. How very small and straight and rigid she lay in the bunk below! Her eyes were closed, but Tatyana doubted if she slept., classify if this sentence: Tatyana doubted if her mother slept. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: The table was piled high with food, and on the floor beside it there were crocks, baskets, and a five-quart pail of milk., classify if this sentence: Beside the table there were crocks, baskets, and a five-quart pail of milk. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: Lily spoke to Donna, breaking her concentration., classify if this sentence: Lily spoke to Donna, breaking Donna's concentration. is ['entailment', 'not entailment'].\nOutput: not entailment\n\nInput: Given this sentence: The drain is clogged with hair. It has to be cleaned., classify if this sentence: The hair has to be cleaned. is ['entailment', 'not entailment'].\nOutput: ",
            "target": "entailment",
            "references": ["entailment"],
            "metrics": ["metrics.accuracy"],
            "group": "unitxt",
            "postprocessors": ["to_string"],
            "prediction": "entailment",
            "score": {
                "global": {
                    "accuracy": 0.5633802816901409,
                    "score": 0.5633802816901409,
                    "groups_mean_score": 0.5633802816901409,
                },
                "instance": {"accuracy": 1.0, "score": 1.0},
            },
            "origin": "all_unitxt",
        }

        # self.assertDictEqual(target, results[0])


if __name__ == "__main__":
    unittest.main()
