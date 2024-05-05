from typing import Dict, List, Tuple

from unitxt.templates import (
    InputOutputTemplate,
    InputOutputTemplateWithCustomTarget,
    KeyValTemplate,
    MultiLabelTemplate,
    MultipleChoiceTemplate,
    MultiReferenceTemplate,
    SpanLabelingJsonTemplate,
    SpanLabelingTemplate,
    Template,
    TemplateFormatKeyError,
    TemplatesDict,
    YesNoTemplate,
)
from unitxt.test_utils.operators import (
    check_operator,
)

from tests.utils import UnitxtTestCase


class TestTemplates(UnitxtTestCase):
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
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
            },
            {
                "inputs": {"text": "hello world"},
                "outputs": {"labels": ["man", "woman", "dog"]},
                "source": "hello world",
                "target": "man, woman, dog",
                "references": ["man, woman, dog"],
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multi_reference_template_without_random_reference(self):
        self._test_multi_reference_template(target="Dan", random_reference=False)

    def test_multi_reference_template_with_random_reference(self):
        self._test_multi_reference_template(target="Dan", random_reference=True)

    def _test_multi_reference_template_with_exception(
        self, references, expected_exception_message: str
    ):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instance = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": references},
        }

        with self.assertRaises(ValueError) as e:
            template.process(instance)
        self.assertEqual(str(e.exception), expected_exception_message)

    def test_multi_reference_template_with_empty_references(self):
        self._test_multi_reference_template_with_exception(
            references=[],
            expected_exception_message="No references found. MultiReferenceTemplate requires at least one reference.",
        )

    def test_multi_reference_template_with_wrong_references_type(self):
        self._test_multi_reference_template_with_exception(
            references=[0, "dkd"],
            expected_exception_message="MultiReferenceTemplate requires references field 'answer' to be List[str]. Got answer<list>: [0, 'dkd']",
        )

    def test_input_output_template_and_standard_template(self):
        template = InputOutputTemplate(
            input_format="This is my text:'{text}'",
            output_format="{label}",
            instruction="Classify sentiment into: {labels}.\n",
            target_prefix="Sentiment is: ",
        )

        inputs = [
            {
                "inputs": {"labels": ["positive", "negative"], "text": "hello world"},
                "outputs": {"label": "positive"},
            },
            {
                "inputs": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "outputs": {"label": "positive"},
            },
            {
                "inputs": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "outputs": {"label": ["positive", "1"]},
            },
        ]

        targets = [
            {
                "inputs": {"labels": ["positive", "negative"], "text": "hello world"},
                "outputs": {"label": "positive"},
                "source": "This is my text:'hello world'",
                "target": "positive",
                "references": ["positive"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
            },
            {
                "inputs": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "outputs": {"label": "positive"},
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive",
                "references": ["positive"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
            },
            {
                "inputs": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "outputs": {"label": ["positive", "1"]},
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive, 1",
                "references": ["positive, 1"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
            },
        ]

        check_operator(template, inputs, targets, tester=self)

        # if "source" and "target" and "instruction_format" and "target_prefix" in instance - instance is not modified
        template = InputOutputTemplate(
            input_format="This is my text:'{text}'",
            output_format="{label}",
        )
        check_operator(template, targets, targets, tester=self)

        err_input_template = InputOutputTemplate(
            input_format="This is my text:'{no_text}'", output_format="{label}"
        )
        with self.assertRaises(TemplateFormatKeyError) as ke:
            err_input_template.process(inputs[0])
        self.assertEqual(
            "\"Available inputs are [labels, text] but InputOutputTemplate.input_format format requires a different ones: 'This is my text:'{no_text}''\"",
            str(ke.exception),
        )

        err_output_template = InputOutputTemplate(
            input_format="This is my text:'{text}'", output_format="{no_label}"
        )
        with self.assertRaises(TemplateFormatKeyError) as ke:
            err_output_template.process(inputs[0])
        self.assertEqual(
            "\"Available outputs are [label] but InputOutputTemplate.output_format format requires a different ones: '{no_label}'\"",
            str(ke.exception),
        )

    def test_input_output_reference_template_and_standard_template(self):
        template = InputOutputTemplateWithCustomTarget(
            input_format="This is my text:'{text}'",
            output_format="{label}",
            instruction="Classify sentiment into: {labels}.\n",
            target_prefix="Sentiment is: ",
            reference="{reference}",
        )

        inputs = [
            {
                "inputs": {"labels": ["positive", "negative"], "text": "hello world"},
                "outputs": {"label": "positive", "reference": "1"},
            },
        ]

        targets = [
            {
                "inputs": {"labels": ["positive", "negative"], "text": "hello world"},
                "outputs": {"label": "positive", "reference": "1"},
                "source": "This is my text:'hello world'",
                "target": "positive",
                "references": ["1"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
            },
        ]

        check_operator(template, inputs, targets, tester=self)

        with self.assertRaises(KeyError):
            template.outputs_to_target_and_references(
                outputs={"label": "positive", "references": "1"}
            )

        class ToCoverTemplate(Template):
            def inputs_to_source(self, inputs: Dict[str, object]) -> Tuple[str, str]:
                ret = super().inputs_to_source(inputs)
                return (ret, ret)

            def outputs_to_target_and_references(
                self, outputs: Dict[str, object]
            ) -> Tuple[str, List[str]]:
                return super().outputs_to_target_and_references(outputs)

        to_cover_template = ToCoverTemplate()
        to_cover_template.inputs_to_source({"a": 1})
        to_cover_template.outputs_to_target_and_references({"a": 1})

        class ToCoverTemplatesDict(TemplatesDict):
            def verify(self):
                super().verify()

        to_cover_templates_dict = ToCoverTemplatesDict()
        to_cover_templates_dict.verify()

    def test_yes_no_template_process_input(self):
        """Test the processing of the input of a YesNoTemplate."""
        template = YesNoTemplate(
            input_format="Is {text} of {class}?",
            class_field="class",
            label_field="labels",
        )

        processed_input_to_inputs = {
            "Is text_a of news?": {"text": "text_a", "class": ["news"]},
            "Is text_b of news?": {"text": "text_b", "class": ["news"]},
        }
        for expected_processed_input, inputs in processed_input_to_inputs.items():
            processed = template.inputs_to_source(inputs)
            self.assertEqual(expected_processed_input, processed)

    def test_yes_no_template_process_input_missing_input_field(self):
        """Test the processing of the input of a YesNoTemplate when one of the fields required in the input is missing. Expect that an exception is thrown."""
        input_format = "Expecting field {class} in input."
        template = YesNoTemplate(
            input_format=input_format, class_field="class", label_field=""
        )
        with self.assertRaises(TemplateFormatKeyError) as cm:
            wrong_field_name = "wrong_field_name"
            template.inputs_to_source(inputs={wrong_field_name: ["news"]})
        self.assertEqual(
            "\"Available inputs are [wrong_field_name] but YesNoTemplate.input_format format requires a different ones: 'Expecting field {class} in input.'\"",
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
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
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
                "instruction": "",
                "target_prefix": "",
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template(self):
        enumerators = ["capitals", "lowercase", "numbers", "roman"]
        firsts = ["A", "a", "1", "I"]
        seconds = ["B", "b", "2", "II"]
        for enumerator, first, second in zip(enumerators, firsts, seconds):
            template = MultipleChoiceTemplate(
                input_format="Text: {text}, Choices: {choices}.", enumerator=enumerator
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
                    "outputs": {
                        "choices": choices,
                        "label": 0,
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                },
                {
                    "inputs": {"choices": choices, "text": "example A"},
                    "outputs": {
                        "choices": choices,
                        "label": "False",
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                },
                {
                    "inputs": {"choices": ["True", "small"], "text": "example A"},
                    "outputs": {
                        "choices": ["True", "small"],
                        "label": "small",
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. small.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                },
            ]

            check_operator(template, inputs, targets, tester=self)

        # check error and more options, to code cover additional lines
        template = MultipleChoiceTemplate(
            input_format="Text: {no_text}, Choices: {no_choices}.",
            postprocessors=["post1", "post2"],
        )

        with self.assertRaises(ValueError) as ve:
            check_operator(template, inputs, targets, tester=self)
        self.assertEqual(
            "Error processing instance '0' from stream 'test' in MultipleChoiceTemplate due to: \"Available inputs are [numerals, choices, text] but MultipleChoiceTemplate.input_format format requires a different ones: 'Text: {no_text}, Choices: {no_choices}.'\"",
            str(ve.exception),
        )

        self.assertListEqual(["post1", "post2"], template.get_postprocessors())

    def test_multiple_choice_template_with_shuffle(self):
        enumerators = ["capitals", "lowercase", "numbers", "roman"]
        firsts = ["A", "a", "1", "I"]
        seconds = ["B", "b", "2", "II"]
        temp = "temp"
        for enumerator, first, second in zip(enumerators, firsts, seconds):
            template = MultipleChoiceTemplate(
                input_format="Text: {text}, Choices: {choices}.",
                enumerator=enumerator,
                shuffle_choices=True,
            )

            inputs = [
                {
                    "inputs": {"choices": ["True", "False"], "text": "example A"},
                    "outputs": {"choices": ["True", "False"], "label": 0},
                },
                {
                    "inputs": {"choices": ["True", "False"], "text": "example A"},
                    "outputs": {"choices": ["True", "False"], "label": "False"},
                },
                {
                    "inputs": {"choices": ["True", temp], "text": "example A"},
                    "outputs": {"choices": ["True", temp], "label": temp},
                },
            ]

            targets = [
                {
                    "inputs": {"choices": ["True", "False"], "text": "example A"},
                    "outputs": {
                        "choices": ["True", "False"],
                        "label": 0,
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                },
                {
                    "inputs": {"choices": ["True", "False"], "text": "example A"},
                    "outputs": {
                        "choices": ["True", "False"],
                        "label": 1,
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                },
                {
                    "inputs": {"choices": [temp, "True"], "text": "example A"},
                    "outputs": {
                        "choices": [temp, "True"],
                        "label": 0,
                        "options": [f"{first}", f"{second}"],
                    },
                    "source": f"Text: example A, Choices: {first}. {temp}, {second}. True.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                },
            ]

            check_operator(template, inputs, targets, tester=self)

        # check error and more options, to code cover additional lines
        template = MultipleChoiceTemplate(
            input_format="Text: {no_text}, Choices: {no_choices}.",
            postprocessors=["post1", "post2"],
        )

        with self.assertRaises(ValueError) as ve:
            check_operator(template, inputs, targets, tester=self)
        self.assertEqual(
            "Error processing instance '0' from stream 'test' in MultipleChoiceTemplate due to: \"Available inputs are [numerals, choices, text] but MultipleChoiceTemplate.input_format format requires a different ones: 'Text: {no_text}, Choices: {no_choices}.'\"",
            str(ve.exception),
        )

        self.assertListEqual(["post1", "post2"], template.get_postprocessors())

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

    def test_render_template(self):
        instance = {"inputs": {"text": "was so bad"}, "outputs": {"label": "negative"}}
        template = InputOutputTemplate(
            input_format='This is my sentence: "{text}"', output_format="{label}"
        )

        result = template.process(instance)
        target = {
            "inputs": {"text": "was so bad"},
            "outputs": {"label": "negative"},
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "",
            "target_prefix": "",
        }
        self.assertDictEqual(result, target)

    def test_render_multi_reference_template(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instance = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
        }

        result = template.process(instance)
        target = {
            "inputs": {"text": "who was he?"},
            "outputs": {"answer": ["Dan", "Yossi"]},
            "source": "This is my sentence: who was he?",
            "target": "Dan",
            "references": ["Dan", "Yossi"],
            "instruction": "",
            "target_prefix": "",
        }
        self.assertDictEqual(result, target)
