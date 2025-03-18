import unitxt
from datasets import load_dataset
from evaluate import load
from unitxt.hf_utils import (
    UnitxtVersionsConflictError,
    _verify_versions,
    get_missing_imports,
)

from tests.utils import UnitxtTestCase


class HFTests(UnitxtTestCase):
    def test_dataset_imports(self):
        missing_imports = get_missing_imports(
            unitxt.dataset_file, exclude=["dataset", "__init__", "api"]
        )
        self.assertEqual(missing_imports, [])

    def test_metric_imports(self):
        missing_imports = get_missing_imports(
            unitxt.metric_file,
            exclude=[
                "metric",
                "metric_utils",
                "__init__",
                "dataset",
                "dataset_utils",
                "api",
            ],
        )
        self.assertEqual(missing_imports, [])

    def test_dataset_load(self):
        dataset = load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0",
            trust_remote_code=True,
            download_mode="force_redownload",
        )
        str(dataset["train"][0])

    def test_metric_load(self):
        load(unitxt.metric_file)

    def test_version_conflicts(self):
        with self.assertRaises(UnitxtVersionsConflictError):
            _verify_versions("test", "0.0.0", "1.1.1")
