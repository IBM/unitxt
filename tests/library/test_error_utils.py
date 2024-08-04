from unitxt.error_utils import DOCUMENTATION_ADDING_TASK, UnitxtError, UnitxtWarning

from tests.utils import UnitxtTestCase


class TestErrorUtils(UnitxtTestCase):
    def test_error_no_additional_info(self):
        with self.assertRaises(UnitxtError) as e:
            raise UnitxtError("This should fail")
        self.assertEqual(str(e.exception), "This should fail")

    def test_error_with_additional_info(self):
        with self.assertRaises(UnitxtError) as e:
            raise UnitxtError("This should fail", DOCUMENTATION_ADDING_TASK)
        self.assertEqual(
            str(e.exception),
            "This should fail\nFor more information: see https://www.unitxt.ai/en/latest//docs/adding_task.html \n",
        )

    def test_warning_no_additional_info(self):
        UnitxtWarning("This should fail")

    def test_warning_with_additional_info(self):
        UnitxtWarning("This should fail", DOCUMENTATION_ADDING_TASK)
