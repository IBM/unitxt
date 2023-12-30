import unittest

from src.unitxt.text_utils import (
    camel_to_snake_case,
    is_camel_case,
    is_snake_case,
    split_words,
)


class TestTextUtils(unittest.TestCase):
    def test_split_words(self):
        test_cases = [
            ("example1", ["example", "1"]),
            ("exampleOne", ["example", "One"]),
            ("123example456", ["123", "example", "456"]),
            ("happyDay", ["happy", "Day"]),
            ("thisIsATest", ["this", "Is", "A", "Test"]),
            ("TestAI2023", ["Test", "AI", "2023"]),
            ("stringWith1Number", ["string", "With", "1", "Number"]),
            ("camelCaseExample", ["camel", "Case", "Example"]),
            ("snake_case_example", ["snake", "case", "example"]),
            ("snake_case2example3", ["snake", "case", "2", "example", "3"]),
            ("kebab-case-example", ["kebab", "case", "example"]),
            ("kebab-case2example3", ["kebab", "case", "2", "example", "3"]),
            ("PascalCaseExample", ["Pascal", "Case", "Example"]),
            ("Title Case Example", ["Title", "Case", "Example"]),
            ("Mixed1Example_case", ["Mixed", "1", "Example", "case"]),
            ("Mixed2Example-case", ["Mixed", "2", "Example", "case"]),
            ("Mixed3_Example-case", ["Mixed", "3", "Example", "case"]),
            ("UPPERCASEEXAMPLE", ["UPPERCASEEXAMPLE"]),
            ("lowercaseexample", ["lowercaseexample"]),
            ("mixedUPanddown", ["mixed", "U", "Panddown"]),
        ]

        for i, (input_string, expected_output) in enumerate(test_cases, 1):
            with self.subTest(i=i):
                self.assertEqual(split_words(input_string), expected_output)

    def test_is_camel_case(self):
        is_camel_case_test_cases = [
            ("isCamelCase", False),
            ("notCamelCase", False),
            ("camelCase", False),
            ("Notcamelcase", True),
            ("camel_Case", False),
            ("camelCase123", False),
            ("camelcase", False),
            ("CAMELCASE", True),
            ("camel-case", False),
            ("HFLoader", True),
        ]

        for i, (input_string, expected_output) in enumerate(
            is_camel_case_test_cases, 1
        ):
            with self.subTest(i=i):
                self.assertEqual(is_camel_case(input_string), expected_output)

    def test_is_snake_case(self):
        is_snake_case_test_cases = [
            ("is_snake_case", True),
            ("Not_snake_case", False),
            ("snake_case", True),
            ("snake_Case", False),
            ("Snakecase", False),
            ("snake-case", False),
            ("snake_case123", True),
            ("123snake_case", True),
            ("snakecase", True),
        ]

        for i, (input_string, expected_output) in enumerate(
            is_snake_case_test_cases, 1
        ):
            with self.subTest(i=i):
                self.assertEqual(is_snake_case(input_string), expected_output)

    def test_camel_to_snake_case(self):
        camel_to_snake_case_test_cases = [
            ("camelToSnake", "camel_to_snake"),
            ("CamelToSnake", "camel_to_snake"),
            ("CamelToSnakeCase", "camel_to_snake_case"),
            ("camelToSnakeCase123", "camel_to_snake_case123"),
            ("123CamelToSnakeCase", "123_camel_to_snake_case"),
            # ("camelTo_Snake_Case", "camel_to__snake__case"), #TODO: Fix this
            # ("camelTo-Snake-Case", "camel_to-_snake-_case"), #TODO: Fix this
            ("camelToSnakeCASE", "camel_to_snake_case"),
            ("CAMELToSnakeCase", "camel_to_snake_case"),
            ("camelToSNAKECase", "camel_to_snake_case"),
            ("HFLoader", "hf_loader"),
        ]

        for i, (input_string, expected_output) in enumerate(
            camel_to_snake_case_test_cases, 1
        ):
            with self.subTest(i=i):
                self.assertEqual(camel_to_snake_case(input_string), expected_output)
