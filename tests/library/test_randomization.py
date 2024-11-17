import unittest
from tempfile import TemporaryDirectory

import unitxt
from datasets import load_dataset
from unitxt.logging_utils import get_logger

from tests.utils import UnitxtTestCase

logger = get_logger()


class TestExamples(UnitxtTestCase):
    def test_dataset_is_deterministic_after_loading_other_dataset(self):
        logger.info("Loading wnli- first time")
        wnli_1_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        logger.info("Loading squad")
        load_dataset(
            unitxt.dataset_file,
            "card=cards.rte,template_card_index=0,num_demos=5,demos_pool_size=100",
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        logger.info("Loading wnli- second time")
        wnli_2_dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
            trust_remote_code=True,
            download_mode="force_redownload",
        )

        self.assertDictEqual(wnli_1_dataset["train"][0], wnli_2_dataset["train"][0])

    def normalize(self, s):
        return " ".join(s.split())

    def test_dataset_is_deterministic_after_augmentation(self):
        logger.info("Loading wnli- first time")
        with TemporaryDirectory() as temp_dir:
            wnli_1_dataset = load_dataset(
                unitxt.dataset_file,
                "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100",
                trust_remote_code=True,
                download_mode="force_redownload",
                cache_dir=temp_dir,
            )
        logger.info("Loading wnli- second time with augmentation")
        with TemporaryDirectory() as temp_dir:
            wnli_2_dataset = load_dataset(
                unitxt.dataset_file,
                "card=cards.wnli,template_card_index=0,num_demos=5,demos_pool_size=100,augmenter=augmenters.image.grey_scale",
                trust_remote_code=True,
                download_mode="force_redownload",
                cache_dir=temp_dir,
            )
        self.maxDiff = None
        for split, i in [("train", 0), ("train", 1), ("test", 0), ("test", 1)]:
            self.assertEqual(
                self.normalize(wnli_1_dataset[split][i]["source"]),
                self.normalize(wnli_2_dataset[split][i]["source"]),
            )


if __name__ == "__main__":
    unittest.main()
