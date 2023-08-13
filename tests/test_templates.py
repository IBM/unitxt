import unittest

from src.unitxt.processors import RegexParser
from src.unitxt.templates import (
    AutoInputOutputTemplate,
    InputOutputTemplate,
    SpanLabelingTemplate,
)
from src.unitxt.test_utils.metrics import apply_metric
from src.unitxt.artifact import fetch_artifact


from src.unitxt.test_utils.catalog import register_local_catalog_for_tests

register_local_catalog_for_tests()

class TestTemplates(unittest.TestCase):
    def test_span_labeling_template_escaping(self):
        
        
        parser, _ = fetch_artifact("processors.to_span_label_pairs")
        
        template = SpanLabelingTemplate()

        inputs = [
            {
                "spans_starts": [0, 19, 41],
                "spans_ends": [10, 27, 48],
                "labels": ["PER", "LOC", "ORG"],
                "text": "John,: Doe is from New York and works at Goo:gle.",
            },
            {
                "spans_starts": [],
                "spans_ends": [],
                "labels": [],
                "text": "John,: Doe is from New York and works at Goo:gle.",
            }
        ]

        processed_targets = [
            "John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG", "None"
        ]

        parsed_targets = [
            [("John\\,\\: Doe", "PER"), ("New York", "LOC"), ("Goo\\:gle", "ORG")], []
        ]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)
            
            
    
    def test_span_labeling_template_one_entity_escaping(self):
        
        
        parser, _ = fetch_artifact("processors.to_span_label_pairs_surface_only")
        
        template = SpanLabelingTemplate(labels_support=['PER'], span_label_format="{span}")

        inputs = [
            {
                "spans_starts": [0, 19, 41],
                "spans_ends": [10, 27, 48],
                "labels": ["PER", "PER", "ORG"],
                "text": "John,: Doe is from New York and works at Goo:gle.",
            },
            {
                "spans_starts": [],
                "spans_ends": [],
                "labels": [],
                "text": "John,: Doe is from New York and works at Goo:gle.",
            }
        ]

        processed_targets = [
            "John\,\: Doe, New York", "None"
        ]

        parsed_targets = [
            [("John\\,\\: Doe", ""), ("New York", "")], []
        ]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)
