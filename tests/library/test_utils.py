from unitxt.utils import (
    is_module_available,
    is_package_installed,
    remove_numerics_and_quoted_texts,
)

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

    def test_remove_numerics_and_quoted_texts(self):
        test_cases = [
            ("This is a string with numbers 1234", "This is a string with numbers "),
            (
                "This string contains a float 123.45 in it",
                "This string contains a float  in it",
            ),
            (
                "This string contains a 'quoted string' here",
                "This string contains a  here",
            ),
            (
                'This string contains a "double quoted string" here',
                "This string contains a  here",
            ),
            (
                '''This string contains a """triple quoted string""" here''',
                "This string contains a  here",
            ),
            (
                '''Here are some numbers 1234 and floats 123.45, and strings 'single' "double" """triple""" ''',
                "Here are some numbers  and floats , and strings    ",
            ),
            (
                "This string contains no numbers or quoted strings",
                "This string contains no numbers or quoted strings",
            ),
        ]

        for i, (input_str, expected_output) in enumerate(test_cases, 1):
            with self.subTest(i=i):
                result = remove_numerics_and_quoted_texts(input_str)
                self.assertEqual(result, expected_output)
