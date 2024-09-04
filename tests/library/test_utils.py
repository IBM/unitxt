from unitxt.utils import is_module_available, is_package_installed

from tests.utils import UnitxtTestCase


class TestUtils(UnitxtTestCase):
    def test_is_package_installed_true(self):
        self.assertTrue(is_package_installed("datasets"))

    def test_is_package_installed_false(self):
        self.assertFalse(is_package_installed("some-non-existent-package-name"))

    def test_is_module_available_true(self):
        self.assertTrue(is_module_available("collections"))

    def test_is_module_available_false(self):
        self.assertFalse(is_module_available("some_non_existent_module"))
