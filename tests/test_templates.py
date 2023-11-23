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


class TestTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

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
        """
        Test the processing of the input of a YesNoTemplate.
        """
        template = YesNoTemplate(input_format="Is {text} of {class}?", class_field="class", label_field="labels")

        proccessed_input_to_inputs = {
            "Is text_a of news?": {"text": "text_a", "class": ["news"]},
            "Is text_b of news?": {"text": "text_b", "class": ["news"]},
        }
        for expected_processed_input, inputs in proccessed_input_to_inputs.items():
            processed = template.process_inputs(inputs)
            self.assertEqual(expected_processed_input, processed)

    def test_yes_no_template_process_input_missing_input_field(self):
        """
        Test the processing of the input of a YesNoTemplate when one of the fields required in the
        input is missing. Expect that an exception is thrown.
        """
        input_format = "Expecting field {class} in input."
        template = YesNoTemplate(input_format=input_format, class_field="class", label_field="")
        with self.assertRaises(RuntimeError) as cm:
            wrong_field_name = "wrong_field_name"
            template.process_inputs(inputs={wrong_field_name: ["news"]})
        self.assertEquals(
            f"Available inputs are ['{wrong_field_name}'] but input format requires a different one: {input_format}",
            str(cm.exception),
        )

    def test_yes_no_template_process_output(self):
        """
        Test the processing of the output of a YesNoTemplate.
        """
        label_field = "labels"
        class_field = "class"
        yes_answer = "y"
        no_answer = "n"
        template = YesNoTemplate(
            input_format="",
            class_field=class_field,
            label_field=label_field,
            yes_answer=yes_answer,
            no_answer=no_answer,
        )

        processed_output_to_outputs = {
            no_answer: {label_field: ["sports"], class_field: ["news"]},
            yes_answer: {label_field: ["news"], class_field: ["news"]},
            yes_answer: {label_field: ["news", "sports"], class_field: ["news"]},
        }
        for expected_processed_output, outputs in processed_output_to_outputs.items():
            processed = template.process_outputs(outputs)
            self.assertEqual(expected_processed_output, processed)

    def test_yes_no_template_process_output_missing_fields(self):
        """
        Test the processing of the output of a YesNoTemplate, when the label_field or the
        class_field values are missing from the output.
        """
        label_field = "labels"
        class_field = "class"
        template = YesNoTemplate(input_format="", class_field=class_field, label_field=label_field)

        with self.assertRaises(RuntimeError) as cm:
            outputs = {class_field: ["news"]}
            template.process_outputs(outputs=outputs)
        self.assertEquals(
            f"Available outputs are {list(outputs.keys())}, missing required label field: '{label_field}'.",
            str(cm.exception),
        )

        with self.assertRaises(RuntimeError) as cm:
            outputs = {label_field: ["news", "sports"]}
            template.process_outputs(outputs=outputs)
        self.assertEquals(
            f"Available outputs are {list(outputs.keys())}, missing required class field: '{class_field}'.",
            str(cm.exception),
        )

    def test_yes_no_template_process_output_wrong_value_in_label_field(self):
        """
        Test the processing of the output of a YesNoTemplate, when the label_field
        contains incorrect values.
        """

        def _test_with_wrong_labels_value(wrong_labels_value):
            template = YesNoTemplate(input_format="", class_field="", label_field="labels")
            with self.assertRaises(RuntimeError) as cm:
                template.process_outputs(outputs={"labels": wrong_labels_value})
            self.assertEquals(
                f"Unexpected value for gold_class_names: '{wrong_labels_value}'. Expected a non-empty list.",
                str(cm.exception),
            )

        _test_with_wrong_labels_value(wrong_labels_value=[])  # list of labels values should not be empty
        _test_with_wrong_labels_value(wrong_labels_value="non list value is an error")

    def test_yes_no_template_process_output_wrong_value_in_class_field(self):
        """
        Test the processing of the output of a YesNoTemplate, when the class_field
        contains incorrect values.
        """

        def _test_with_wrong_class_value(wrong_class_value):
            label_field = "labels"
            class_field = "class"
            template = YesNoTemplate(input_format="", class_field=class_field, label_field=label_field)
            with self.assertRaises(RuntimeError) as cm:
                template.process_outputs(
                    outputs={
                        label_field: ["news"],
                        class_field: wrong_class_value,
                    }
                )
            self.assertEquals(
                f"Unexpected value for queried_class_names: '{wrong_class_value}'. Expected a list with one item.",
                str(cm.exception),
            )

        _test_with_wrong_class_value(wrong_class_value=[])  # list of class values should not be empty
        _test_with_wrong_class_value(wrong_class_value="non list value is an error")
        _test_with_wrong_class_value(wrong_class_value=["list with", "two or more items is an error"])

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
