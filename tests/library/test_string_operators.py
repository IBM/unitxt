from unitxt.string_operators import Join, RegexSplit, Split, TokensSplit
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
