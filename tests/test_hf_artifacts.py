import unittest

from datasets import load_dataset
from evaluate import load
from src import unitxt
from src.unitxt.hf_utils import get_missing_imports


class HFTests(unittest.TestCase):
    def test_dataset_imports(self):
        missing_imports = get_missing_imports(unitxt.dataset_file, exclude=["dataset", "__init__"])
        self.assertEqual(missing_imports, [])

    def test_metric_imports(self):
        missing_imports = get_missing_imports(unitxt.metric_file, exclude=["metric", "__init__", "dataset"])
        self.assertEqual(missing_imports, [])

    def test_dataset_load(self):
        dataset = load_dataset(
            unitxt.dataset_file, "card=cards.wnli,template_item=0", download_mode="force_redownload"
        )

    def test_metric_load(self):
        metric = load(unitxt.metric_file)
