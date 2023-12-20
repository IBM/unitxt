import unittest

from src.unitxt.artifact import fetch_artifact


class TestPostProcessors(unittest.TestCase):
    def test_convert_to_boolean(self):
        parser, _ = fetch_artifact("processors.convert_to_boolean")
        inputs = [
            "that's right",
            "correct",
            "not sure",
            "true",
            "TRUE",
            "false",
            "interesting",
        ]
        targets = ["TRUE", "TRUE", "FALSE", "TRUE", "TRUE", "FALSE", "OTHER"]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_to_span_label_pairs(self):
        parser, _ = fetch_artifact("processors.to_span_label_pairs")
        inputs = [r"John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG", "None"]
        targets = [
            [("John\\,\\: Doe", "PER"), ("New York", "LOC"), ("Goo\\:gle", "ORG")],
            [],
        ]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_to_list_by_comma(self):
        parser, _ = fetch_artifact("processors.to_list_by_comma")
        inputs = ["cat, dog", "man, woman, dog"]
        targets = [["cat", "dog"], ["man", "woman", "dog"]]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_to_span_label_pairs_surface_only(self):
        parser, _ = fetch_artifact("processors.to_span_label_pairs_surface_only")
        inputs = [r"John\,\: Doe, New York", "None"]
        targets = [[("John\\,\\: Doe", ""), ("New York", "")], []]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_load_json(self):
        parser, _ = fetch_artifact("processors.load_json")
        inputs = [
            '{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}',
            "None",
        ]

        targets = [
            {"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]},
            [],
        ]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_dict_of_lists_to_value_key_pairs(self):
        parser, _ = fetch_artifact("processors.dict_of_lists_to_value_key_pairs")
        inputs = [
            {"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]},
            {},
        ]

        targets = [
            [("John,: Doe", "PER"), ("New York", "PER"), ("Goo:gle", "ORG")],
            [],
        ]

        for input, target in zip(inputs, targets):
            parsed = parser.process(input)
            self.assertEqual(target, parsed)

    def test_span_labeling_json_template_errors(self):
        postprocessor1, _ = fetch_artifact("processors.load_json")
        postprocessor2, _ = fetch_artifact(
            "processors.dict_of_lists_to_value_key_pairs"
        )

        predictions = ["{}", '{"d":{"b": "c"}}', '{dll:"dkk"}', '["djje", "djjjd"]']

        post1_targets = [{}, {"d": {"b": "c"}}, [], ["djje", "djjjd"]]
        post2_targets = [[], [("b", "d")], [], []]

        for pred, post_target1, post_target2 in zip(
            predictions, post1_targets, post2_targets
        ):
            post1 = postprocessor1.process(pred)
            self.assertEqual(post1, post_target1)
            post2 = postprocessor2.process(post1)
            self.assertEqual(post2, post_target2)
