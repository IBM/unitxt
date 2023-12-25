import unittest

from src.unitxt.templates import (
    InputOutputTemplate,
    KeyValTemplate,
    MultiLabelTemplate,
    MultipleChoiceTemplate,
    MultiReferenceTemplate,
    SpanLabelingJsonTemplate,
    SpanLabelingTemplate,
    YesNoTemplate,
)
from src.unitxt.test_utils.catalog import register_local_catalog_for_tests
from src.unitxt.test_utils.operators import (
    check_operator,
)


class TestTemplates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_local_catalog_for_tests()

    def test_span_labeling_template_escaping(self):
        template = SpanLabelingTemplate(input_format="{text}")

        inputs = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "LOC", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "LOC", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": r"John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG",
                "references": [r"John\,\: Doe: PER, New York: LOC, Goo\:gle: ORG"],
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": "None",
                "references": ["None"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multi_label_template(self):
        template = MultiLabelTemplate(input_format="{text}")

        inputs = [
            {
                "inputs": {"text": "hello world"},
                "outputs": {"labels": ["cat", "dog"]},
            },
            {
                "inputs": {"text": "hello world"},
                "outputs": {"labels": ["man", "woman", "dog"]},
            },
        ]

        targets = [
            {
                "inputs": {"text": "hello world"},
                "outputs": {"labels": ["cat", "dog"]},
                "source": "hello world",
                "target": "cat, dog",
                "references": ["cat, dog"],
            },
            {
                "inputs": {"text": "hello world"},
                "outputs": {"labels": ["man", "woman", "dog"]},
                "source": "hello world",
                "target": "man, woman, dog",
                "references": ["man, woman, dog"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def _test_multi_reference_template(self, target, random_reference):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}",
            references_field="answer",
            random_reference=random_reference,
        )

        inputs = [
            {
                "inputs": {"text": "who was he?"},
                "outputs": {"answer": ["Dan", "Yossi"]},
            }
        ]

        targets = [
            {
                "inputs": {"text": "who was he?"},
                "outputs": {"answer": ["Dan", "Yossi"]},
                "source": "This is my sentence: who was he?",
                "target": target,
                "references": ["Dan", "Yossi"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multi_reference_template_without_random_reference(self):
        self._test_multi_reference_template(target="Dan", random_reference=False)

    def test_multi_reference_template_with_random_reference(self):
        self._test_multi_reference_template(target="Yossi", random_reference=True)

    def test_multi_reference_template_verify_references_type(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instance = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": [0, "dkd"]},
        }

        with self.assertRaises(ValueError):
            template.process(instance)

    def test_input_output_template(self):
        template = InputOutputTemplate(
            input_format="This is my text:'{text}'", output_format="{label}"
        )

        inputs = [
            {"inputs": {"text": "hello world"}, "outputs": {"label": "positive"}},
            {
                "inputs": {"text": ["hello world\n", "hell"]},
                "outputs": {"label": "positive"},
            },
            {
                "inputs": {"text": ["hello world\n", "hell"]},
                "outputs": {"label": ["positive", "1"]},
            },
        ]

        targets = [
            {
                "source": "This is my text:'hello world'",
                "target": "positive",
                "references": ["positive"],
            },
            {
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive",
                "references": ["positive"],
            },
            {
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive, 1",
                "references": ["positive, 1"],
            },
        ]

        targets = [{**target, **input} for target, input in zip(targets, inputs)]

        check_operator(template, inputs, targets, tester=self)

    def test_yes_no_template_process_input(self):
        """Test the processing of the input of a YesNoTemplate."""
        template = YesNoTemplate(
            input_format="Is {text} of {class}?",
            class_field="class",
            label_field="labels",
        )

        proccessed_input_to_inputs = {
            "Is text_a of news?": {"text": "text_a", "class": ["news"]},
            "Is text_b of news?": {"text": "text_b", "class": ["news"]},
        }
        for expected_processed_input, inputs in proccessed_input_to_inputs.items():
            processed = template.inputs_to_source(inputs)
            self.assertEqual(expected_processed_input, processed)

    def test_yes_no_template_process_input_missing_input_field(self):
        """Test the processing of the input of a YesNoTemplate when one of the fields required in the input is missing. Expect that an exception is thrown."""
        input_format = "Expecting field {class} in input."
        template = YesNoTemplate(
            input_format=input_format, class_field="class", label_field=""
        )
        with self.assertRaises(RuntimeError) as cm:
            wrong_field_name = "wrong_field_name"
            template.inputs_to_source(inputs={wrong_field_name: ["news"]})
        self.assertEqual(
            f"Available inputs are ['{wrong_field_name}'] but input format requires a different one: {input_format}",
            str(cm.exception),
        )

    def test_yes_no_template_process_output(self):
        """Test the processing of the output of a YesNoTemplate."""
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
            yes_answer: {label_field: ["news", "sports"], class_field: ["news"]},
        }
        for expected_processed_output, outputs in processed_output_to_outputs.items():
            processed, references = template.outputs_to_target_and_references(outputs)
            self.assertEqual(expected_processed_output, processed)
            self.assertEqual(references, [expected_processed_output])

    def test_yes_no_template_process_output_missing_fields(self):
        """Test the processing of the output of a YesNoTemplate.

        Test the processing of the output of a YesNoTemplate when the label_field or the class_field values are missing from the output.
        """
        label_field = "labels"
        class_field = "class"
        template = YesNoTemplate(
            input_format="", class_field=class_field, label_field=label_field
        )

        with self.assertRaises(RuntimeError) as cm:
            outputs = {class_field: ["news"]}
            template.outputs_to_target_and_references(outputs=outputs)
        self.assertEqual(
            f"Available outputs are {list(outputs.keys())}, missing required label field: '{label_field}'.",
            str(cm.exception),
        )

        with self.assertRaises(RuntimeError) as cm:
            outputs = {label_field: ["news", "sports"]}
            template.outputs_to_target_and_references(outputs=outputs)
        self.assertEqual(
            f"Available outputs are {list(outputs.keys())}, missing required class field: '{class_field}'.",
            str(cm.exception),
        )

    def test_yes_no_template_process_output_wrong_value_in_label_field(self):
        """Test the processing of the output of a YesNoTemplate, when the label_field contains incorrect values."""

        def _test_with_wrong_labels_value(wrong_labels_value):
            template = YesNoTemplate(
                input_format="", class_field="", label_field="labels"
            )
            with self.assertRaises(RuntimeError) as cm:
                template.outputs_to_target_and_references(
                    outputs={"labels": wrong_labels_value}
                )
            self.assertEqual(
                f"Unexpected value for gold_class_names: '{wrong_labels_value}'. Expected a non-empty list.",
                str(cm.exception),
            )

        _test_with_wrong_labels_value(
            wrong_labels_value=[]
        )  # list of labels values should not be empty
        _test_with_wrong_labels_value(wrong_labels_value="non list value is an error")

    def test_yes_no_template_process_output_wrong_value_in_class_field(self):
        """Test the processing of the output of a YesNoTemplate, when the class_field contains incorrect values."""

        def _test_with_wrong_class_value(wrong_class_value):
            label_field = "labels"
            class_field = "class"
            template = YesNoTemplate(
                input_format="", class_field=class_field, label_field=label_field
            )
            with self.assertRaises(RuntimeError) as cm:
                template.outputs_to_target_and_references(
                    outputs={
                        label_field: ["news"],
                        class_field: wrong_class_value,
                    }
                )
            self.assertEqual(
                f"Unexpected value for queried_class_names: '{wrong_class_value}'. Expected a list with one item.",
                str(cm.exception),
            )

        _test_with_wrong_class_value(
            wrong_class_value=[]
        )  # list of class values should not be empty
        _test_with_wrong_class_value(wrong_class_value="non list value is an error")
        _test_with_wrong_class_value(
            wrong_class_value=["list with", "two or more items is an error"]
        )

    def test_span_labeling_template_one_entity_escaping(self):
        template = SpanLabelingTemplate(
            input_format="{text}", labels_support=["PER"], span_label_format="{span}"
        )

        inputs = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": r"John\,\: Doe, New York",
                "references": [r"John\,\: Doe, New York"],
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": "None",
                "references": ["None"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_span_labeling_json_template(self):
        template = SpanLabelingJsonTemplate(input_format="{text}")

        inputs = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "inputs": {"text": "John,: Doe is from New York and works at Goo:gle."},
                "outputs": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": '{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}',
                "references": [
                    '{"PER": ["John,: Doe", "New York"], "ORG": ["Goo:gle"]}'
                ],
            },
            {
                "inputs": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "outputs": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "source": "John,: Doe is from New York and works at Goo:gle.",
                "target": "None",
                "references": ["None"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
        )

        choices = ["True", "False"]
        inputs = [
            {
                "inputs": {"choices": choices, "text": "example A"},
                "outputs": {"choices": choices, "label": 0},
            },
            {
                "inputs": {"choices": choices, "text": "example A"},
                "outputs": {"choices": choices, "label": "False"},
            },
            {
                "inputs": {"choices": ["True", "small"], "text": "example A"},
                "outputs": {"choices": ["True", "small"], "label": "small"},
            },
        ]

        targets = [
            {
                "inputs": {"choices": choices, "text": "example A"},
                "outputs": {"choices": choices, "label": 0, "options": ["A", "B"]},
                "source": "Text: example A, Choices: A. True, B. False.",
                "target": "A",
                "references": ["A"],
            },
            {
                "inputs": {"choices": choices, "text": "example A"},
                "outputs": {
                    "choices": choices,
                    "label": "False",
                    "options": ["A", "B"],
                },
                "source": "Text: example A, Choices: A. True, B. False.",
                "target": "B",
                "references": ["B"],
            },
            {
                "inputs": {"choices": ["True", "small"], "text": "example A"},
                "outputs": {
                    "choices": ["True", "small"],
                    "label": "small",
                    "options": ["A", "B"],
                },
                "source": "Text: example A, Choices: A. True, B. small.",
                "target": "B",
                "references": ["B"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_key_val_template_simple(self):
        template = KeyValTemplate()

        dic = {"hello": "world", "str_list": ["djjd", "djjd"]}

        result = template.process_dict(
            dic, key_val_sep=": ", pairs_sep=", ", use_keys=True
        )
        target = "hello: world, str_list: djjd, djjd"
        self.assertEqual(result, target)

    def test_key_val_template_int_list(self):
        template = KeyValTemplate()

        dic = {"hello": "world", "int_list": [0, 1]}

        result = template.process_dict(
            dic, key_val_sep=": ", pairs_sep=", ", use_keys=True
        )
        target = "hello: world, int_list: 0, 1"
        self.assertEqual(result, target)
