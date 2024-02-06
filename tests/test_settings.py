import os
import unittest

from src.unitxt.settings_utils import Settings, get_settings


class TestSettings(unittest.TestCase):
    def test_singleton(self):
        settings1 = Settings()
        settings2 = Settings()

        self.assertEqual(settings1, settings2)

    def test_singleton_assignment(self):
        settings1 = Settings()
        settings2 = Settings()
        settings1.test = "test"

        self.assertEqual(settings2.test, "test")

    def test_singleton_assignment_with_get_settings(self):
        settings1 = Settings()
        settings2 = get_settings()
        settings1.test = "test"

        self.assertEqual(settings2.test, "test")

    def test_key_creation(self):
        settings = Settings()
        settings.test = "text"

        self.assertEqual(settings.test_key, "UNITXT_TEST")

    def test_env_var_override(self):
        settings = Settings()
        settings.test = "text"
        os.environ[settings.test_key] = "not_text"
        self.assertEqual(settings.test, "not_text")
