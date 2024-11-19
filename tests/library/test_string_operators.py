import re

from unitxt.string_operators import (
    Join,
    RegexReplace,
    RegexSplit,
    Split,
    TokensSlice,
    TokensSplit,
)
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestStringOperators(UnitxtTestCase):
    def test_split(self):
        operator = Split(field="text", by=",")
        inputs = [{"text": "kk,ll"}]
        targets = [{"text": ["kk", "ll"]}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_regex_split(self):
        operator = RegexSplit(field="text", by=r"([\s,!?]+)")
        inputs = [{"text": "Hello, world! How are you?"}]
        targets = [
            {
                "text": [
                    "Hello",
                    ", ",
                    "world",
                    "! ",
                    "How",
                    " ",
                    "are",
                    " ",
                    "you",
                    "?",
                    "",
                ]
            }
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_tokens_split(self):
        operator = TokensSplit(field="text", model="bert-base-uncased")
        inputs = [{"text": "Here's an example sentence for tokenization."}]
        targets = [
            {
                "text": [
                    "here",
                    "'",
                    "s",
                    "an",
                    "example",
                    "sentence",
                    "for",
                    "token",
                    "##ization",
                    ".",
                ]
            }
        ]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_join(self):
        operator = Join(field="text", by=",")
        inputs = [{"text": ["kk", "ll"]}]
        targets = [{"text": "kk,ll"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_tokens_slice(self):
        operator = TokensSlice(field="text", model="gpt2", stop=1)
        inputs = [{"text": "hello world"}]
        targets = [{"text": "hello"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_basic_regex_replacement(self):
        operator = RegexReplace(
            field="text", pattern=r"^\s*This is my answer:\s*", replacement=""
        )
        inputs = [{"text": "   \nThis is my answer:    Here is the response."}]
        targets = [{"text": "Here is the response."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_no_match_regex_replacement(self):
        operator = RegexReplace(
            field="text", pattern=r"^\s*This is my answer:\s*", replacement=""
        )
        inputs = [{"text": "Some unrelated text."}]
        targets = [{"text": "Some unrelated text."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_pattern_with_special_characters_regex_replacement(self):
        operator = RegexReplace(
            field="text", pattern=r"^\s*\*\*\*Special Phrase:\s*", replacement=""
        )
        inputs = [{"text": "***Special Phrase:    Some other text."}]
        targets = [{"text": "Some other text."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_multiple_lines_regex_replacement(self):
        operator = RegexReplace(
            field="text", pattern=r"^\s*This is my answer:\s*", replacement=""
        )
        inputs = [{"text": "   \nThis is my answer:\nHere is another line."}]
        targets = [{"text": "Here is another line."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_dynamic_pattern_regex_replacement(self):
        dynamic_phrase = "Dynamic Test Phrase"
        escaped_phrase = re.escape(dynamic_phrase)
        operator = RegexReplace(
            field="text", pattern=rf"^\s*{escaped_phrase}\s*", replacement=""
        )
        inputs = [{"text": "   Dynamic Test Phrase    Extra text."}]
        targets = [{"text": "Extra text."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_dynamic_pattern_with_special_characters_regex_replacement(self):
        dynamic_phrase = "Hello (world)*"
        escaped_phrase = re.escape(dynamic_phrase)
        operator = RegexReplace(
            field="text", pattern=rf"^\s*{escaped_phrase}\s*", replacement=""
        )
        inputs = [{"text": "   Hello (world)*    Extra text."}]
        targets = [{"text": "Extra text."}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
