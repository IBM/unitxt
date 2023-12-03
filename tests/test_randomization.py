import logging
import unittest

from datasets import load_dataset

from src import unitxt
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    def test_dataset_is_deterministic_after_loading_other_dataset(self):
        logging.info("Loading wnli- first time")
        wnli_1_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        logging.info("Loading squad")
        load_dataset(
            unitxt.dataset_file,
            "card=cards.rte,template_card_index=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        logging.info("Loading wnli- second time")
        wnli_2_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )

        self.assertDictEqual(wnli_1_dataset["train"][0], wnli_2_dataset["train"][0])

    def normalize(self, s):
        return " ".join(s.split())

    def test_dataset_is_deterministic_after_augmentation(self):
        logging.info("Loading wnli- first time")
        wnli_1_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            download_mode="force_redownload",
        )
        logging.info("Loading wnli- second time with augmentation")
        wnli_2_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100,augmentor=augmentors.augment_whitespace_model_input",
            download_mode="force_redownload",
        )
        self.maxDiff = None
        for split, i in [("train", 0), ("train", 1), ("test", 0), ("test", 1)]:
            self.assertEqual(
                self.normalize(wnli_1_dataset[split][i]["source"]),
                self.normalize(wnli_2_dataset[split][i]["source"]),
            )


if __name__ == "__main__":
    unittest.main()
