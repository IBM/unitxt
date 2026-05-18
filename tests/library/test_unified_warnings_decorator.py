import unittest
import warnings

from unitxt.deprecation_utils import warn_on_call


class TestWarnOnCall(unittest.TestCase):
    def test_warning_on_instance_creation(self):
        @warn_on_call(UserWarning, "Class object initialized!")
        class TestClass:
            def __init__(self, name):
                self.name = name

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            obj = TestClass("Initialized_object")

        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        self.assertTrue(len(user_warnings) > 0)
        self.assertEqual(str(user_warnings[0].message), "Class object initialized!")
        self.assertEqual(obj.name, "Initialized_object")

    def test_warning_called_on_instance_creation(self):
        @warn_on_call(UserWarning, "Class object initialization warning.")
        class MockedClass:
            def __init__(self, name):
                self.name = name

        with unittest.mock.patch("warnings.warn") as mock_warn:
            mock_warn.assert_not_called()

            MockedClass("Instance")
            mock_warn.assert_called_once_with(
                "Class object initialization warning.", UserWarning, stacklevel=2
            )


if __name__ == "__main__":
    unittest.main()
