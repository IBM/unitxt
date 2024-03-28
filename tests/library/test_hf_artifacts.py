from datasets import load_dataset
from evaluate import load

from src import unitxt
from src.unitxt.hf_utils import (
    UnitxtVersionsConflictError,
    get_missing_imports,
    verify_versions_compatibility,
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
        load_dataset(
            unitxt.dataset_file,
            "card=cards.wnli,template_card_index=0",
            download_mode="force_redownload",
        )

    def test_metric_load(self):
        load(unitxt.metric_file)

    def test_version_conflicts_lower(self):
        with self.assertRaises(UnitxtVersionsConflictError):
            verify_versions_compatibility("test", "0.0.0")

    def test_version_conflicts_higher(self):
        with self.assertRaises(UnitxtVersionsConflictError):
            verify_versions_compatibility("test", "10000000000.0.0")
