from unitxt.span_lableing_operators import IobExtractor
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestSpanLabelingOperator(UnitxtTestCase):
    def test_iob_extractor(self):
        self.maxDiff = None
        operator = IobExtractor(
            labels=["Person", "Organization", "Location"],
            begin_labels=["B-PER", "B-ORG", "B-LOC"],
            inside_labels=["I-PER", "I-ORG", "I-LOC"],
            outside_label="O",
        )

        inputs = [
            {
                "text": "John Doe works at OpenAI",
                "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
                "tokens": ["John", "Doe", "works", "at", "OpenAI"],
            },
            {
                "text": "This is test",
                "labels": ["O", "O", "O"],
                "tokens": ["This", "is", "test"],
            },
            {
                "text": "John Doe from London.",
                "labels": ["B-PER", "I-PER", "O", "B-LOC", "O"],
                "tokens": ["John", "Doe", "from", "London", "."],
            },
            {
                "text": "John London is OpenAI",
                "labels": ["B-PER", "B-LOC", "O", "B-ORG"],
                "tokens": ["John", "London", "is", "OpenAI"],
            },
        ]

        targets = [
            {
                "text": "John Doe works at OpenAI",
                "labels": ["B-PER", "I-PER", "O", "B-ORG", "I-ORG"],
                "tokens": ["John", "Doe", "works", "at", "OpenAI"],
                "spans": [
                    {"start": 0, "end": 8, "text": "John Doe", "label": "Person"},
                    {
                        "start": 15,
                        "end": 24,
                        "text": "at OpenAI",
                        "label": "Organization",
                    },
                ],
            },
            {
                "text": "This is test",
                "labels": ["O", "O", "O"],
                "tokens": ["This", "is", "test"],
                "spans": [],
            },
            {
                "text": "John Doe from London.",
                "labels": ["B-PER", "I-PER", "O", "B-LOC", "O"],
                "tokens": ["John", "Doe", "from", "London", "."],
                "spans": [
                    {"start": 0, "end": 8, "text": "John Doe", "label": "Person"},
                    {"start": 14, "end": 20, "text": "London", "label": "Location"},
                ],
            },
            {
                "text": "John London is OpenAI",
                "labels": ["B-PER", "B-LOC", "O", "B-ORG"],
                "tokens": ["John", "London", "is", "OpenAI"],
                "spans": [
                    {"start": 0, "end": 4, "text": "John", "label": "Person"},
                    {"start": 5, "end": 11, "text": "London", "label": "Location"},
                    {"start": 15, "end": 21, "text": "OpenAI", "label": "Organization"},
                ],
            },
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
