import os

from src.unitxt.settings_utils import Settings, get_settings
from tests.utils import UnitxtTestCase


class TestSettings(UnitxtTestCase):
    def test_singleton(self):
        settings1 = Settings()
        settings2 = Settings()

        self.assertEqual(settings1, settings2)

    def test_singleton_assignment(self):
        settings1 = Settings()
        settings2 = Settings()
        settings1.test1 = "test1"

        self.assertEqual(settings2.test1, "test1")

    def test_typed_assignment(self):
        settings = Settings()
        settings.test_bool_assignment = (bool, "False")
        settings.test_bool_assignment2 = (bool, "True")
        settings.test_int_assignment = (int, 1)
        settings.test_int_assignment = "0"
        settings.test_float_assignment = (float, None)
        settings.test_float_assignment = "0.01"

        self.assertEqual(settings.test_bool_assignment, False)
        self.assertEqual(settings.test_bool_assignment2, True)
        self.assertEqual(settings.test_int_assignment, 0)
        self.assertEqual(settings.test_float_assignment, 0.01)

        with self.assertRaises(ValueError):
            settings.test_bool_assignment = "True1"

        with self.assertRaises(ValueError):
            settings.test_str_assignment = (str, None)

    def test_singleton_assignment_with_get_settings(self):
        settings1 = Settings()
        settings2 = get_settings()
        settings1.test2 = "test2"
        settings1.test2 = "test3"
        settings1.test2 = "test2"

        self.assertEqual(settings2.test2, "test2")

    def test_key_creation(self):
        settings = Settings()
        settings.test3 = "text3"

        self.assertEqual(settings.test3_key, "UNITXT_TEST3")

        with self.assertRaises(AttributeError):
            settings.test3_key = "dummy"

    def test_env_var_override(self):
        settings = Settings()
        settings.test_env = "text_env"
        os.environ[settings.test_env_key] = "not_text_env"
        self.assertEqual(settings.test_env, "not_text_env")

    def test_env_var_typing(self):
        settings = Settings()
        settings.test_bool_var = (bool, True)
        settings.test_int_var = (int, 5)
        settings.test_float_var = (float, 0.7)

        os.environ[settings.test_bool_var_key] = "False"
        self.assertEqual(settings.test_bool_var, False)

        os.environ[settings.test_bool_var_key] = "TRUE"
        with self.assertRaises(ValueError):
            _ = settings.test_bool_var

        os.environ[settings.test_int_var_key] = "12"
        self.assertEqual(settings.test_int_var, 12)

        os.environ[settings.test_float_var_key] = "5.82"
        self.assertEqual(settings.test_float_var, 5.82)
