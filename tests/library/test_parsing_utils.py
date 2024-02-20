from src.unitxt.parsing_utils import (
    parse_key_equals_value_string_to_dict,
    separate_inside_and_outside_square_brackets,
)
from tests.utils import UnitxtTestCase


class TestParsingUtils(UnitxtTestCase):
    # Tests for parse_key_equals_value_string_to_dict
    def test_parse_key_equals_value_string_to_dict_simple_query(self):
        query = "name=John Doe,age=30,height=5.8"
        expected = {"name": "John Doe", "age": 30, "height": 5.8}
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

    def test_parse_key_equals_value_string_to_dict_with_spaces(self):
        query = "first name=Jane Doe, last name=Doe, country=USA, balance=100.50"
        expected = {
            "first name": "Jane Doe",
            "last name": "Doe",
            "country": "USA",
            "balance": 100.50,
        }
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

    def test_parse_key_equals_value_string_to_dict_empty_value(self):
        query = "username=admin,password="
        with self.assertRaises(ValueError):
            parse_key_equals_value_string_to_dict(query)

    def test_parse_key_equals_value_string_to_dict_missing_value(self):
        query = "username=admin,role"
        with self.assertRaises(ValueError):
            parse_key_equals_value_string_to_dict(query)

    def test_parse_key_equals_value_string_to_dict_numeric_values(self):
        query = "year=2020,score=9.5,count=10"
        expected = {"year": 2020, "score": 9.5, "count": 10}
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

    def test_parse_key_equals_value_string_to_dict_illegal_format(self):
        query = "malformed"
        with self.assertRaises(ValueError):
            parse_key_equals_value_string_to_dict(query)

    # Additional test for handling booleans
    def test_parse_key_equals_value_string_to_dict_boolean_values(self):
        query = "is_valid=true,has_errors=false"
        expected = {"is_valid": "true", "has_errors": "false"}
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

    def test_base_structure(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets("text"), ("text", None)
        )

    def test_valid_structure(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets("before[inside]"),
            ("before", "inside"),
        )

    def test_valid_nested_structure(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets(
                "before[inside_before[inside_inside]]"
            ),
            ("before", "inside_before[inside_inside]"),
        )

    def test_valid_nested_structure_with_broken_structre(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets(
                "before[inside_before[inside_inside]"
            ),
            ("before", "inside_before[inside_inside"),
        )

    def test_valid_nested_structure_with_broken_structre_inside(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets(
                "before[inside_a]between[inside_b]"
            ),
            ("before", "inside_a]between[inside_b"),
        )

    def test_valid_empty_inside(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets("before[]"), ("before", "")
        )

    def test_illegal_text_following_brackets(self):
        with self.assertRaisesRegex(
            ValueError,
            "Illegal structure: text follows after the closing square bracket.",
        ):
            separate_inside_and_outside_square_brackets("before[inside]after")

    def test_illegal_unmatched_left_bracket(self):
        with self.assertRaisesRegex(
            ValueError, "Illegal structure: unmatched square brackets."
        ):
            separate_inside_and_outside_square_brackets("before[inside")

    def test_illegal_unmatched_right_bracket(self):
        with self.assertRaisesRegex(
            ValueError, "Illegal structure: unmatched square brackets."
        ):
            separate_inside_and_outside_square_brackets("before]inside")

    def test_illegal_extra_characters_after_closing_bracket(self):
        with self.assertRaisesRegex(
            ValueError,
            "Illegal structure: text follows after the closing square bracket.",
        ):
            separate_inside_and_outside_square_brackets("[inside]extra")

    def test_illegal_nested_brackets(self):
        with self.assertRaisesRegex(
            ValueError,
            "Illegal structure: text follows after the closing square bracket.",
        ):
            separate_inside_and_outside_square_brackets("before[inside[nested]]after")

    def test_illegal_multiple_bracket_pairs(self):
        with self.assertRaisesRegex(
            ValueError,
            "Illegal structure: text follows after the closing square bracket.",
        ):
            separate_inside_and_outside_square_brackets(
                "before[inside]middle[another]after"
            )
