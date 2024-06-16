import unitxt
from datasets import load_dataset
from unitxt.text_utils import print_dict

from tests.utils import UnitxtTestCase


class TestExamples(UnitxtTestCase):
    def test_evaluate(self):
        import evaluate
        import unitxt
        from unitxt.text_utils import print_dict

        dataset = unitxt.load_dataset("card=cards.wnli,template_card_index=0")

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
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        print_dict(dataset["train"][0])
        # self.assertDictEqual(target, dataset['train'][0])

    def test_full_flow_of_hf(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            trust_remote_code=True,
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
