import unittest

from src.unitxt.artifact import fetch_artifact
from src.unitxt.processors import RegexParser
from src.unitxt.templates import (
    AutoInputOutputTemplate,
    InputOutputTemplate,
    KeyValTemplate,
    MultiLabelTemplate,
    SpanLabelingJsonTemplate,
    SpanLabelingTemplate,
    YesNoTemplate,
)
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests
from src.unitxt.test_utils.metrics import apply_metric

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
            },
        ]

        processed_targets = ["John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG", "None"]

        parsed_targets = [[("John\\,\\: Doe", "PER"), ("New York", "LOC"), ("Goo\\:gle", "ORG")], []]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)

    def test_multi_label_template(self):
        parser, _ = fetch_artifact("processors.to_list_by_comma")

        template = MultiLabelTemplate()

        inputs = [
            {"labels": ["cat", "dog"]},
            {"labels": ["man", "woman", "dog"]},
        ]

        processed_targets = ["cat, dog", "man, woman, dog"]

        parsed_targets = [["cat", "dog"], ["man", "woman", "dog"]]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)

    def test_input_output_template(self):
        parser, _ = fetch_artifact("processors.to_string_stripped")

        template = InputOutputTemplate(output_format="{labels}")

        inputs = [
            {"labels": ["cat"]},
            {"labels": [" man"]},
            {"labels": ["dog\n"]},
        ]

        processed_targets = ["cat", " man", "dog\n"]

        parsed_targets = ["cat", "man", "dog"]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)

    def test_yes_no_template_process_input(self):
        template = YesNoTemplate(input_format="Is {text} of {class_name}?", class_name="news", label_field="labels")

        proccessed_input_to_inputs = {
            "Is text_a of news?": {"text": "text_a"},
            "Is text_b of news?": {"text": "text_b"},
        }
        for expected_processed_input, inputs in proccessed_input_to_inputs.items():
            processed = template.process_inputs(inputs)
            self.assertEqual(expected_processed_input, processed)

    def test_yes_no_template_process_input_missing_input_field(self):
        input_format = "Expecting field {text} in input."
        template = YesNoTemplate(input_format=input_format, class_name="", label_field="")
        with self.assertRaises(KeyError) as cm:
            wrong_field_name = "wrong_field_name"
            template.process_inputs(inputs={wrong_field_name: "text_a"})
            self.assertEquals(
                f"Available inputs are {wrong_field_name} but input format requires a different one: {input_format}",
                str(cm.exception),
            )

    def test_yes_no_template_process_output(self):
        label_field = "labels"
        class_name = "news"
        yes_answer = "y"
        no_answer = "n"
        template = YesNoTemplate(
            input_format="",
            class_name=class_name,
            label_field=label_field,
            yes_answer=yes_answer,
            no_answer=no_answer,
        )

        processed_output_to_outputs = {
            no_answer: {label_field: ["sports"]},
            yes_answer: {label_field: [class_name]},
            yes_answer: {label_field: [class_name, "sports"]},
        }
        for expected_processed_output, outputs in processed_output_to_outputs.items():
            processed = template.process_outputs(outputs)
            self.assertEqual(expected_processed_output, processed)

    def test_yes_no_template_process_output_missing_label_field(self):
        label_field = "labels"
        template = YesNoTemplate(input_format="", class_name="", label_field=label_field)
        with self.assertRaises(KeyError) as cm:
            outputs = {}
            template.process_outputs(outputs=outputs)
            self.assertEquals(
                f"Available outputs are {outputs.keys()}, but required label field is: '{label_field}'.",
                str(cm.exception),
            )

    def test_yes_no_template_process_output_wrong_value_in_label_field(self):
        template = YesNoTemplate(input_format="", class_name="", label_field="labels")
        with self.assertRaises(RuntimeError) as cm:
            gold_class_names = []
            template.process_outputs(outputs={"labels": gold_class_names})
            self.assertEquals(
                f"Unexpected value for gold_class_names: '{gold_class_names}'. Expected a non-empty list.",
                str(cm.exception),
            )
        with self.assertRaises(RuntimeError) as cm:
            gold_class_names = "non list value"
            template.process_outputs(outputs={"labels": gold_class_names})
            self.assertEquals(
                f"Unexpected value for gold_class_names: '{gold_class_names}'. Expected a non-empty list.",
                str(cm.exception),
            )

    def test_span_labeling_template_one_entity_escaping(self):
        parser, _ = fetch_artifact("processors.to_span_label_pairs_surface_only")

        template = SpanLabelingTemplate(labels_support=["PER"], span_label_format="{span}")

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
            },
        ]

        processed_targets = ["John\,\: Doe, New York", "None"]

        parsed_targets = [[("John\\,\\: Doe", ""), ("New York", "")], []]

        for input, processed_target, parsed_target in zip(inputs, processed_targets, parsed_targets):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            parsed = parser.process(processed)
            self.assertEqual(parsed, parsed_target)

    def test_span_labeling_json_template(self):
        postprocessor1, _ = fetch_artifact("processors.load_json")
        postprocessor2, _ = fetch_artifact("processors.dict_of_lists_to_value_key_pairs")

        template = SpanLabelingJsonTemplate()

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
            },
        ]

        processed_targets = ['{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}', "None"]

        post1_targets = [{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}, []]
        post2_targets = [[("John,: Doe", "PER"), ("New York", "PER"), ("Goo:gle", "ORG")], []]

        for input, processed_target, post_target1, post_target2 in zip(
            inputs, processed_targets, post1_targets, post2_targets
        ):
            processed = template.process_outputs(input)
            self.assertEqual(processed, processed_target)
            post1 = postprocessor1.process(processed)
            self.assertEqual(post1, post_target1)
            post2 = postprocessor2.process(post1)
            self.assertEqual(post2, post_target2)

    def test_span_labeling_json_template_errors(self):
        postprocessor1, _ = fetch_artifact("processors.load_json")
        postprocessor2, _ = fetch_artifact("processors.dict_of_lists_to_value_key_pairs")

        predictions = ["{}", '{"d":{"b": "c"}}', '{dll:"dkk"}', '["djje", "djjjd"]']

        post1_targets = [{}, {"d": {"b": "c"}}, [], ["djje", "djjjd"]]
        post2_targets = [[], [("b", "d")], [], []]

        for pred, post_target1, post_target2 in zip(predictions, post1_targets, post2_targets):
            post1 = postprocessor1.process(pred)
            self.assertEqual(post1, post_target1)
            post2 = postprocessor2.process(post1)
            self.assertEqual(post2, post_target2)

    def test_key_val_template_simple(self):
        template = KeyValTemplate()

        dic = {"hello": "world", "str_list": ["djjd", "djjd"]}

        result = template.process_dict(dic, key_val_sep=": ", pairs_sep=", ", use_keys=True)
        target = "hello: world, str_list: djjd, djjd"
        self.assertEqual(result, target)

    def test_key_val_template_int_list(self):
        template = KeyValTemplate()

        dic = {"hello": "world", "int_list": [0, 1]}

        result = template.process_dict(dic, key_val_sep=": ", pairs_sep=", ", use_keys=True)
        target = "hello: world, int_list: 0, 1"
        self.assertEqual(result, target)
