from unitxt.parsing_utils import (
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

        # hyphen is allowed inside names, also at the beginning
        query = "name=John-Doe,-age=30,--height=5.8"
        expected = {"name": "John-Doe", "-age": 30, "--height": 5.8}
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

        # constants: True, False, None
        query = "name=John-Doe,-age=30,--height=5.8,wife=None,happy=False,rich=True"
        expected = {
            "name": "John-Doe",
            "-age": 30,
            "--height": 5.8,
            "wife": None,
            "happy": False,
            "rich": True,
        }
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

        query = "year=2020,score=9.5,count=10, balance=-10"
        expected = {"year": 2020, "score": 9.5, "count": 10, "balance": -10}
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

        query = "year=2020,score=9.5,count=10, balance=-10.302"
        expected = {"year": 2020, "score": 9.5, "count": 10, "balance": -10.302}
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

    def test_parse_key_equals_value_string_to_dict_lists(self):
        query = "year=2020,score=[9.5,nine and a half],count=10"
        expected = {"year": 2020, "score": [9.5, "nine and a half"], "count": 10}
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

        query = "year=2020,score=[9.5,nine, and, a half],count=10"
        expected = {"year": 2020, "score": [9.5, "nine", "and", "a half"], "count": 10}
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

        query = "year=2020,score=[9.5, artifactname[init=[nine, and, a, half], anotherinner=True]],count=10"
        expected = {
            "year": 2020,
            "score": [
                9.5,
                "artifactname[init=[nine, and, a, half], anotherinner=True]",
            ],
            "count": 10,
        }
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

    def test_parse_key_equals_value_string_to_dict_illegal_format(self):
        query = "malformed"
        with self.assertRaises(ValueError):
            parse_key_equals_value_string_to_dict(query)

        query = "name=John Doe,age=30,height=5.8 ]and left overs"
        with self.assertRaises(ValueError):
            parse_key_equals_value_string_to_dict(query)

    # Additional test for handling booleans
    def test_parse_key_equals_value_string_to_dict_boolean_values(self):
        query = "is_valid=true,has_errors=false"
        expected = {"is_valid": "true", "has_errors": "false"}
        self.assertEqual(parse_key_equals_value_string_to_dict(query), expected)

        query = "is_valid=True,has_errors=False"
        expected = {"is_valid": True, "has_errors": False}
        self.assertEqual(expected, parse_key_equals_value_string_to_dict(query))

    def test_base_structure(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets("text"), ("text", None)
        )

    def test_valid_structure(self):
        self.assertEqual(
            separate_inside_and_outside_square_brackets("before[inside=2]"),
            ("before", {"inside": 2}),
        )

    def test_valid_nested_structure(self):
        self.assertEqual(
            ("before", {"inside_before": "inside_inside[iii=vvv]"}),
            separate_inside_and_outside_square_brackets(
                "before[inside_before=inside_inside[iii=vvv]]"
            ),
        )

    def test_valid_empty_inside(self):
        self.assertEqual(
            ("before", {}), separate_inside_and_outside_square_brackets("before[]")
        )

    def test_illegal_text_following_brackets(self):
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets("before[inside]after")
        self.assertEqual("malformed assignment in: inside]after", str(ve.exception))
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets("before[ins=ide]after")
        self.assertEqual(
            "malformed end of query: excessive text following the ] that closes the overwrites in: 'before[ins=ide]after'",
            str(ve.exception),
        )

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
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets("before[ins=ide]extra")
        self.assertEqual(
            "malformed end of query: excessive text following the ] that closes the overwrites in: 'before[ins=ide]extra'",
            str(ve.exception),
        )

    def test_illegal_nested_brackets(self):
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets("before[ins=ide[nest=ed]]after")
        self.assertEqual(
            "malformed end of query: excessive text following the ] that closes the overwrites in: 'before[ins=ide[nest=ed]]after'",
            str(ve.exception),
        )
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets("before[inside[nested]]after")
        self.assertEqual(
            "malformed assignment in: inside[nested]]after", str(ve.exception)
        )

    def test_illegal_multiple_bracket_pairs(self):
        with self.assertRaises(ValueError) as ve:
            separate_inside_and_outside_square_brackets(
                "before[inside]middle[another]after"
            )
        self.assertEqual(
            "malformed assignment in: inside]middle[another]after", str(ve.exception)
        )
