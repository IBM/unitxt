import unitxt
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
            unitxt.dataset_file, exclude=["dataset", "evaluate_cli", "__init__", "api"]
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
                "evaluate_cli",
                "api",
            ],
        )
        self.assertEqual(missing_imports, [])

    def test_metric_load(self):
        load(unitxt.metric_file)

    def test_version_conflicts(self):
        with self.assertRaises(UnitxtVersionsConflictError):
            _verify_versions("test", "0.0.0", "1.1.1")
