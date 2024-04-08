from unitxt.formats import apply_capital_new_line_notation

from tests.utils import UnitxtTestCase


class TestApplyCapitalNewLineNotation(UnitxtTestCase):
    def test_double_new_line_converted_to_single(self):
        self.assertEqual(apply_capital_new_line_notation("bla\n\n\\N"), "bla\n")

    def test_single_new_line_with_capital_converted_to_single(self):
        self.assertEqual(apply_capital_new_line_notation("bla\n\\N"), "bla\n")

    def test_multiple_new_lines_converted_to_empty(self):
        self.assertEqual(apply_capital_new_line_notation("\n\n\n\\N"), "")

    def test_single_capital_new_line_converted_to_empty(self):
        self.assertEqual(apply_capital_new_line_notation("\\N"), "")

    def test_double_capital_new_lines_converted_to_single_new_line(self):
        self.assertEqual(apply_capital_new_line_notation("bla\\N\\N"), "bla\n")

    def test_mixed_capital_and_regular_new_lines_converted_to_single(self):
        self.assertEqual(apply_capital_new_line_notation("bla\\N\n"), "bla\n\n")

    def test_capital_new_line_between_text_converted(self):
        self.assertEqual(apply_capital_new_line_notation("bla\\Nbla1"), "bla\nbla1")

    def test_string_starting_with_capital_new_line_trimmed(self):
        self.assertEqual(apply_capital_new_line_notation("\\Nbla"), "bla")

    def test_string_ending_with_capital_new_line_converted(self):
        self.assertEqual(apply_capital_new_line_notation("bla\\N"), "bla\n")

    def test_sequential_capital_new_lines_reduced_to_single_new_line(self):
        self.assertEqual(apply_capital_new_line_notation("\\N\\N\\N"), "")

    def test_sequential_capital_new_lines_reduced_to_single_new_line2(self):
        self.assertEqual(apply_capital_new_line_notation("\\N\\N\\Nbla"), "bla")

    def test_embedded_capital_new_line_converted(self):
        self.assertEqual(
            apply_capital_new_line_notation("bla\\Nnewline"), "bla\nnewline"
        )

    def test_mixed_new_lines_and_capitals_normalized(self):
        self.assertEqual(
            apply_capital_new_line_notation("Mixed\n\\Newlines\\Nand\\N\nCapitals"),
            "Mixed\newlines\nand\n\nCapitals",
        )

    def test_only_new_line_characters_converted(self):
        self.assertEqual(apply_capital_new_line_notation("\n\\N\n"), "\n")
