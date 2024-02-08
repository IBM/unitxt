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
        settings1.test1 = "test1"

        self.assertEqual(settings2.test1, "test1")

    def test_singleton_assignment_with_get_settings(self):
        settings1 = Settings()
        settings2 = get_settings()
        settings1.test2 = "test2"

        self.assertEqual(settings2.test2, "test2")

    def test_key_creation(self):
        settings = Settings()
        settings.test3 = "text3"

        self.assertEqual(settings.test3_key, "UNITXT_TEST3")

    def test_env_var_override(self):
        settings = Settings()
        settings.test_env = "text_env"
        os.environ[settings.test_env_key] = "not_text_env"
        self.assertEqual(settings.test_env, "not_text_env")
