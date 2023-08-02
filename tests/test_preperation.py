import glob
import os
import unittest

from tests.unitxt_test_case import setup_unitxt_test_env

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
glob_query = os.path.join(project_dir, "prepare", "**", "*.py")
all_prepration_files = glob.glob(glob_query, recursive=True)


@setup_unitxt_test_env
class TestExamples(unittest.TestCase):
    def test_preprations(self):
        print(glob_query)
        print(f"Testing prepration files: {all_prepration_files}")
        for file in all_prepration_files:
            with self.subTest(file=file):
                print(f"Testing prepration file: {file}")
                with open(file, "r") as f:
                    exec(f.read())
                print(f"Testing prepration file: {file} passed")
                self.assertTrue(True)
