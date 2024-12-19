import pandas as pd
from unitxt.text_utils import (
    camel_to_snake_case,
    is_camel_case,
    is_snake_case,
    lines_defining_obj_in_card,
    split_words,
    to_pretty_string,
)

from tests.utils import UnitxtTestCase


class TestTextUtils(UnitxtTestCase):
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

    def test_lines_defining_obj(self):
        with open("prepare/cards/cohere_for_ai.py") as fp:
            all_lines = fp.readlines()
        starting, ending = lines_defining_obj_in_card(
            all_lines=all_lines, obj_name="TaskCard("
        )
        self.assertEqual("        card = TaskCard(\n", all_lines[starting])
        self.assertEqual("        )\n", all_lines[ending])

        starting_desc_in_card, ending_desc_in_card = lines_defining_obj_in_card(
            all_lines=all_lines[starting:ending],
            obj_name="__description__",
        )
        self.assertIn("__description__=(", all_lines[starting + starting_desc_in_card])

        # now test with __description__ that does not open with ( nor ends with a
        # closing ) that is alone in its line
        with open("prepare/cards/numeric_nlg.py") as fp:
            all_lines = fp.readlines()
        starting, ending = lines_defining_obj_in_card(
            all_lines=all_lines, obj_name="TaskCard("
        )
        self.assertEqual("card = TaskCard(\n", all_lines[starting])
        self.assertEqual(")\n", all_lines[ending])

        starting_desc_in_card, ending_desc_in_card = lines_defining_obj_in_card(
            all_lines=all_lines[starting:ending],
            obj_name="__description__",
        )
        self.assertEqual(starting_desc_in_card, ending_desc_in_card)
        self.assertIn("__description__=", all_lines[starting + starting_desc_in_card])

    def test_pretty_string_simple_dict(self):
        data = {"key1": "value1", "key2": 123}
        result = to_pretty_string(data, max_chars=80)
        expected = "key1 (str):\n" "    value1\n" "key2 (int):\n" "    123\n"
        self.assertEqual(result, expected)

    def test_pretty_string_nested_structures(self):
        data = {
            "user": {
                "name": "Bob",
                "stats": {
                    "score": 999,
                    "ranking": [1, 2, 3],
                },
            }
        }

        result = to_pretty_string(data, max_chars=80)
        expected = (
            "user (dict):\n"
            "    name (str):\n"
            "        Bob\n"
            "    stats (dict):\n"
            "        score (int):\n"
            "            999\n"
            "        ranking (list):\n"
            "            [0] (int):\n"
            "                1\n"
            "            [1] (int):\n"
            "                2\n"
            "            [2] (int):\n"
            "                3\n"
        )
        self.assertEqual(result, expected)

    def test_pretty_string_line_wrapping(self):
        long_text = "This is a long line that should be wrapped around multiple times to fit the width."
        data = {"description": long_text}
        result = to_pretty_string(data, max_chars=80)
        # Given the logic, the long line should be wrapped after 76 chars (due to indentation)
        # First line chunk: "This is a long line that should be wrapped around multiple times to fit the "
        # Second line chunk: "width."
        expected = (
            "description (str):\n"
            "    This is a long line that should be wrapped around multiple times to fit the\n"
            "    width.\n"
        )
        self.assertEqual(result, expected)

    def test_pretty_string_list_and_tuple(self):
        data = {"numbers": [10, 20, 30], "coords": (42.0, 23.5)}
        result = to_pretty_string(data, max_chars=80)
        expected = (
            "numbers (list):\n"
            "    [0] (int):\n"
            "        10\n"
            "    [1] (int):\n"
            "        20\n"
            "    [2] (int):\n"
            "        30\n"
            "coords (tuple):\n"
            "    (0) (float):\n"
            "        42.0\n"
            "    (1) (float):\n"
            "        23.5\n"
        )
        self.assertEqual(result, expected)

    def test_pretty_string_dataframe_wrapping(self):
        # Create a DataFrame with many columns and long column names
        df = pd.DataFrame(
            {
                "Short": [1, 2],
                "A_very_long_column_name_to_force_wrapping": [3, 4],
                "Another_extremely_long_column_name_that_will_exceed_width": [5, 6],
                "Yet_another_long_column_to_ensure_wrapping_occurs_properly": [7, 8],
            }
        )

        # We choose a small max_chars to ensure wrapping is triggered
        result = to_pretty_string(df, max_chars=50)
        self.assertEqual(
            result,
            "   Short  \\\n0      1\n1      2\n\n   A_very_long_column_name_to_force_wrapping  \\\n0                                          3\n1                                          4\n\n   Another_extremely_long_column_name_that_will_ex\nceed_width  \\\n0\n         5\n1\n         6\n\n   Yet_another_long_column_to_ensure_wrapping_occu\nrs_properly\n0\n          7\n1\n          8\n",
        )
