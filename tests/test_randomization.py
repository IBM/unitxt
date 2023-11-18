import unittest

from datasets import load_dataset
from src import unitxt
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    def test_dataset_is_deterministic_after_loading_other_dataset(self):
        print("Loading wnli- first time")
        wnli_1_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_item=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        print("Loading squad")
        squad_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.rte,template_item=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        print("Loading wnli- second time")
        wnli_2_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_item=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )

        self.assertDictEqual(wnli_1_dataset["train"][0], wnli_2_dataset["train"][0])


if __name__ == "__main__":
    unittest.main()
