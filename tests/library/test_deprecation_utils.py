import warnings
from unittest.mock import patch

from unitxt.deprecation_utils import DeprecationError, compare_versions, deprecation

from tests.utils import UnitxtTestCase


class EnsureWarnings:
    def __enter__(self):
        self.original_filters = warnings.filters[:]
        warnings.resetwarnings()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.filters = self.original_filters


class PatchConstants:
    def __init__(self, version) -> None:
        self.version = version


class TestDeprecationUtils(UnitxtTestCase):
    def test_compare_versions_equal(self):
        self.assertEqual(compare_versions("1.0.0", "1.0.0"), 0)

    def test_compare_versions_first_greater(self):
        self.assertEqual(compare_versions("1.0.1", "1.0.0"), 1)

    def test_compare_versions_second_greater(self):
        self.assertEqual(compare_versions("1.0.0", "1.0.1"), -1)

    def test_compare_versions_with_different_lengths(self):
        self.assertEqual(compare_versions("1.0", "1.0.0"), 0)
        self.assertEqual(compare_versions("1.0.1", "1.0"), 1)
        self.assertEqual(compare_versions("1.0", "1.0.1"), -1)

    @patch("unitxt.deprecation_utils.constants", PatchConstants(version="1.0.0"))
    def test_deprecation_warning(self):
        @deprecation("1.1.0")
        def some_deprecated_function():
            return "I'm deprecated but not yet obsolete."

        with EnsureWarnings():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = some_deprecated_function()
                self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
                self.assertEqual(result, "I'm deprecated but not yet obsolete.")

    @patch("unitxt.deprecation_utils.constants", PatchConstants(version="2.0.0"))
    def test_deprecation_error(self):
        @deprecation("1.5.0", "use_some_other_function")
        def some_obsolete_function():
            return "I'm obsolete."

        with self.assertRaises(DeprecationError):
            some_obsolete_function()

    @patch("unitxt.deprecation_utils.constants", PatchConstants(version="1.0.0"))
    def test_class_deprecation_warning(self):
        @deprecation("2.0.0", alternative="NewClass")
        class DeprecatedClass:
            def __init__(self):
                pass

            def some_method(self):
                return "method running"

        with EnsureWarnings():
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                obj = DeprecatedClass()
                result = obj.some_method()
                self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
                self.assertEqual(result, "method running")

    @patch("unitxt.deprecation_utils.constants", PatchConstants(version="3.0.0"))
    def test_class_deprecation_error(self):
        @deprecation("2.0.0", alternative="NewClass")
        class DeprecatedClass:
            pass

        with self.assertRaises(DeprecationError):
            DeprecatedClass()

    def test_custom_exception(self):
        with self.assertRaises(DeprecationError):
            raise DeprecationError("This version is no longer supported.")
