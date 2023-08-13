import unittest

from src.unitxt.processors import RegexParser
from src.unitxt.templates import (
    AutoInputOutputTemplate,
    InputOutputTemplate,
    SpanLabelingTemplate,
)
from src.unitxt.test_utils.metrics import apply_metric

# parse string like "1:hlle, 2:world" list of tuples using regex
regex = r"\s*((?:[^,:\\]|\\.)+?)\s*:\s*((?:[^,:\\]|\\.)+?)\s*(?=,|$)"

# test regext parser
parser = RegexParser(regex=regex)


class TestTemplates(unittest.TestCase):
    def test_span_labeling_template_escaping(self):
        template = SpanLabelingTemplate()

        inputs = [
            {
                "spans_starts": [0, 19, 41],
                "spans_ends": [10, 27, 48],
                "labels": ["PER", "LOC", "ORG"],
                "text": "John,: Doe is from New York and works at Goo:gle.",
            }
        ]

        processed_targets = [
            "John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG",
        ]

        parsed_targets = [
            [("John\\,\\: Doe", "PER"), ("New York", "LOC"), ("Goo\\:gle", "ORG")],
        ]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)
