from typing import Dict, List, Tuple

from unitxt.dataclass import RequiredFieldError
from unitxt.error_utils import UnitxtError
from unitxt.templates import (
    ApplyRandomTemplate,
    ApplySingleTemplate,
    InputOutputTemplate,
    InputOutputTemplateWithCustomTarget,
    KeyValTemplate,
    MultiLabelTemplate,
    MultipleChoiceTemplate,
    MultiReferenceTemplate,
    OutputQuantizingTemplate,
    SpanLabelingJsonTemplate,
    SpanLabelingTemplate,
    Template,
    TemplateFormatKeyError,
    TemplatesDict,
    YesNoTemplate,
)
from unitxt.test_utils.operators import (
    check_operator,
    check_operator_exception,
)

from tests.utils import UnitxtTestCase


class TestTemplates(UnitxtTestCase):
    def test_span_labeling_template_escaping(self):
        template = SpanLabelingTemplate(input_format="{text}")

        inputs = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "LOC", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
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
                "postprocessors": ["processors.to_span_label_pairs"],
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
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
                "postprocessors": ["processors.to_span_label_pairs"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multi_label_template(self):
        template = MultiLabelTemplate(input_format="{text}")

        inputs = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
            },
        ]

        targets = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
                "source": "hello world",
                "target": "cat, dog",
                "references": ["cat, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
                "source": "hello world",
                "target": "man, woman, dog",
                "references": ["man, woman, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_output_quantizing_template(self):
        template = OutputQuantizingTemplate(
            input_format="{text}", output_format="{label}", quantum=0.5
        )

        inputs = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"label": 3.4},
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"label": 1},
            },
        ]

        targets = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"label": 3.4},
                "source": "hello world",
                "target": "3.5",
                "references": ["3.5"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"label": 1},
                "source": "hello world",
                "target": "1.0",
                "references": ["1.0"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_apply_single_template(self):
        base_template = MultiLabelTemplate(input_format="{text}")
        template = ApplySingleTemplate(template=base_template, demos_field="demos")

        inputs = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["cat", "dog"]},
                    }
                ],
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["man", "woman", "dog"]},
                    }
                ],
            },
        ]

        targets = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
                "source": "hello world",
                "target": "cat, dog",
                "references": ["cat, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["cat", "dog"]},
                        "source": "hello world",
                        "target": "cat, dog",
                        "references": ["cat, dog"],
                        "instruction": "",
                        "target_prefix": "",
                        "postprocessors": ["processors.to_list_by_comma"],
                    }
                ],
                "recipe_metadata": {"template": base_template},
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
                "source": "hello world",
                "target": "man, woman, dog",
                "references": ["man, woman, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["man", "woman", "dog"]},
                        "source": "hello world",
                        "target": "man, woman, dog",
                        "references": ["man, woman, dog"],
                        "instruction": "",
                        "target_prefix": "",
                        "postprocessors": ["processors.to_list_by_comma"],
                    }
                ],
                "recipe_metadata": {"template": base_template},
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_apply_random_template(self):
        temp1 = MultiLabelTemplate(input_format="temp1 {text}")
        temp2 = MultiLabelTemplate(input_format="temp2 {text}")
        template = ApplyRandomTemplate(
            templates=[
                temp1,
                temp2,
            ],
            demos_field="demos",
        )

        inputs = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["cat", "dog"]},
                    }
                ],
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["man", "woman", "dog"]},
                    }
                ],
            },
        ]

        targets = [
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["cat", "dog"]},
                "source": "temp2 hello world",
                "target": "cat, dog",
                "references": ["cat, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["cat", "dog"]},
                        "source": "temp2 hello world",
                        "target": "cat, dog",
                        "references": ["cat, dog"],
                        "instruction": "",
                        "target_prefix": "",
                        "postprocessors": ["processors.to_list_by_comma"],
                    }
                ],
                "recipe_metadata": {"template": temp2},
            },
            {
                "input_fields": {"text": "hello world"},
                "reference_fields": {"labels": ["man", "woman", "dog"]},
                "source": "temp1 hello world",
                "target": "man, woman, dog",
                "references": ["man, woman, dog"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_list_by_comma"],
                "demos": [
                    {
                        "input_fields": {"text": "hello world"},
                        "reference_fields": {"labels": ["man", "woman", "dog"]},
                        "source": "temp1 hello world",
                        "target": "man, woman, dog",
                        "references": ["man, woman, dog"],
                        "instruction": "",
                        "target_prefix": "",
                        "postprocessors": ["processors.to_list_by_comma"],
                    }
                ],
                "recipe_metadata": {"template": temp1},
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def _test_multi_reference_template(
        self, target, random_reference, references=("Dan", "Yossi")
    ):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}",
            references_field="answer",
            random_reference=random_reference,
        )

        inputs = [
            {
                "input_fields": {"text": "who was he?"},
                "reference_fields": {"answer": list(references)},
            }
        ]

        targets = [
            {
                "input_fields": {"text": "who was he?"},
                "reference_fields": {"answer": list(references)},
                "source": "This is my sentence: who was he?",
                "target": target,
                "references": [str(r) for r in references],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
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
            "input_fields": {"text": "who was he?"},
            "reference_fields": {"answer": references},
        }

        with self.assertRaises(UnitxtError) as e:
            template.process(instance)
        self.assertIn(expected_exception_message, str(e.exception))

    # We now allow multi_reference templates without references
    # to more simply support LLM as Judges
    # def test_multi_reference_template_with_empty_references(self):
    #    self._test_multi_reference_template_with_exception(
    #        references=[],
    #        expected_exception_message="No references found in field 'answer' of instance. MultiReferenceTemplate requires at least one reference.",
    #    )

    def test_multi_reference_template_with_wrong_references_type(self):
        self._test_multi_reference_template(
            target="0", references=[0, "dkd"], random_reference=False
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
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": "hello world",
                },
                "reference_fields": {"label": "positive"},
            },
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "reference_fields": {"label": "positive"},
            },
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "reference_fields": {"label": ["positive", "1"]},
            },
        ]

        targets = [
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": "hello world",
                },
                "reference_fields": {"label": "positive"},
                "source": "This is my text:'hello world'",
                "target": "positive",
                "references": ["positive"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
                "postprocessors": ["processors.to_string_stripped"],
            },
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "reference_fields": {"label": "positive"},
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive",
                "references": ["positive"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
                "postprocessors": ["processors.to_string_stripped"],
            },
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": ["hello world\n", "hell"],
                },
                "reference_fields": {"label": ["positive", "1"]},
                "source": "This is my text:'hello world\n, hell'",
                "target": "positive, 1",
                "references": ["positive, 1"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
                "postprocessors": ["processors.to_string_stripped"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

        # if "source" and "target" and "instruction_format" and "target_prefix" in instance - instance is not modified
        template = InputOutputTemplate(
            input_format="This is my text: {text}",
            output_format="{label}",
        )
        check_operator(template, targets, targets, tester=self)

        err_input_template = InputOutputTemplate(
            input_format="This is my text: {no_text}", output_format="{label}"
        )
        with self.assertRaises(TemplateFormatKeyError) as ke:
            err_input_template.process(inputs[0])
        self.assertIn(
            "Available input fields are [labels, text] but InputOutputTemplate.input_format format requires a different ones: 'This is my text: {no_text}'",
            str(ke.exception),
        )

        err_output_template = InputOutputTemplate(
            input_format="This is my text: {text}", output_format="{no_label}"
        )
        with self.assertRaises(TemplateFormatKeyError) as ke:
            err_output_template.process(inputs[0])
        self.assertIn(
            "Available reference fields are [label] but InputOutputTemplate.output_format format requires a different ones: '{no_label}'",
            str(ke.exception),
        )

        err_output_template = InputOutputTemplate(
            input_format="This is my text: {text}"
        )
        with self.assertRaises(UnitxtError) as ke:
            err_output_template.process(inputs[0])
        self.assertIn(
            "Required field 'output_format' of class InputOutputTemplate not set in InputOutputTemplate",
            str(ke.exception),
        )

        with self.assertRaises(RequiredFieldError) as ke:
            err_output_template = InputOutputTemplate(output_format="{label}")
            err_output_template.process(inputs[0])
        self.assertIn(
            "Required field 'input_format' of class InputFormatTemplate not set in InputOutputTemplate",
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
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": "hello world",
                },
                "reference_fields": {"label": "positive", "reference": "1"},
            },
        ]

        targets = [
            {
                "input_fields": {
                    "labels": ["positive", "negative"],
                    "text": "hello world",
                },
                "reference_fields": {"label": "positive", "reference": "1"},
                "source": "This is my text:'hello world'",
                "target": "positive",
                "references": ["1"],
                "instruction": "Classify sentiment into: positive, negative.\n",
                "target_prefix": "Sentiment is: ",
                "postprocessors": ["processors.to_string_stripped"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

        with self.assertRaises(UnitxtError):
            template.reference_fields_to_target_and_references(
                reference_fields={"label": "positive", "references": "1"}
            )

        class ToCoverTemplate(Template):
            def input_fields_to_source(
                self, inputs: Dict[str, object]
            ) -> Tuple[str, str]:
                ret = super().input_fields_to_source(inputs)
                return (ret, ret)

            def reference_fields_to_target_and_references(
                self, outputs: Dict[str, object]
            ) -> Tuple[str, List[str]]:
                return super().reference_fields_to_target_and_references(outputs)

        to_cover_template = ToCoverTemplate()
        to_cover_template.input_fields_to_source({"a": 1})
        to_cover_template.reference_fields_to_target_and_references({"a": 1})

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
            "Is text_a of news?": {"text": "text_a", "class": "news"},
            "Is text_b of news?": {"text": "text_b", "class": "news"},
        }
        for expected_processed_input, inputs in processed_input_to_inputs.items():
            processed = template.input_fields_to_source(inputs)
            self.assertEqual(expected_processed_input, processed)

    def test_yes_no_template_process_input_missing_input_field(self):
        """Test the processing of the input of a YesNoTemplate when one of the fields required in the input is missing. Expect that an exception is thrown."""
        input_format = "Expecting field {class} in input."
        template = YesNoTemplate(
            input_format=input_format, class_field="class", label_field=""
        )
        with self.assertRaises(TemplateFormatKeyError) as cm:
            wrong_field_name = "wrong_field_name"
            template.input_fields_to_source(input_fields={wrong_field_name: ["news"]})
        self.assertIn(
            "Available input fields are [wrong_field_name] but YesNoTemplate.input_format format requires a different ones: 'Expecting field {class} in input.'",
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
            no_answer: {label_field: ["sports"], class_field: "news"},
            yes_answer: {label_field: ["news", "sports"], class_field: "news"},
        }
        for expected_processed_output, outputs in processed_output_to_outputs.items():
            processed, references = template.reference_fields_to_target_and_references(
                outputs
            )
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

        with self.assertRaises(UnitxtError) as cm:
            outputs = {class_field: "news"}
            template.reference_fields_to_target_and_references(reference_fields=outputs)
        self.assertIn(
            f"Available reference_fields are {list(outputs.keys())}, missing required label field: '{label_field}'.",
            str(cm.exception),
        )

        with self.assertRaises(UnitxtError) as cm:
            outputs = {label_field: ["news", "sports"]}
            template.reference_fields_to_target_and_references(reference_fields=outputs)
        self.assertIn(
            f"Available reference_fields are {list(outputs.keys())}, missing required class field: '{class_field}'.",
            str(cm.exception),
        )

    def test_yes_no_template_process_output_wrong_value_in_label_field(self):
        """Test the processing of the output of a YesNoTemplate, when the label_field contains incorrect values."""

        def _test_with_wrong_labels_value(wrong_labels_value):
            template = YesNoTemplate(
                input_format="", class_field="", label_field="labels"
            )
            with self.assertRaises(UnitxtError) as cm:
                template.reference_fields_to_target_and_references(
                    reference_fields={"labels": wrong_labels_value}
                )
            self.assertIn(
                f"Unexpected value for gold_class_names: '{wrong_labels_value}'. Expecting a list.",
                str(cm.exception),
            )

        _test_with_wrong_labels_value(wrong_labels_value="non list value is an error")

    def test_yes_no_template_process_output_wrong_value_in_class_field(self):
        """Test the processing of the output of a YesNoTemplate, when the class_field contains incorrect values."""

        def _test_with_wrong_class_value(wrong_class_value):
            label_field = "labels"
            class_field = "class"
            template = YesNoTemplate(
                input_format="", class_field=class_field, label_field=label_field
            )
            with self.assertRaises(UnitxtError) as cm:
                template.reference_fields_to_target_and_references(
                    reference_fields={
                        label_field: ["news"],
                        class_field: wrong_class_value,
                    }
                )
            self.assertIn(
                f"Unexpected value for queried_class_names: '{wrong_class_value}'. Expected a string.",
                str(cm.exception),
            )

        _test_with_wrong_class_value(
            wrong_class_value=None
        )  # list of class values should not be empty
        _test_with_wrong_class_value(wrong_class_value=["list is a wrong value"])

    def test_span_labeling_template_one_entity_escaping(self):
        template = SpanLabelingTemplate(
            input_format="{text}", labels_support=["PER"], span_label_format="{span}"
        )

        inputs = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
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
                "postprocessors": ["processors.to_span_label_pairs"],
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
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
                "postprocessors": ["processors.to_span_label_pairs"],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_span_labeling_json_template(self):
        template = SpanLabelingJsonTemplate(input_format="{text}")

        inputs = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
                    "spans_starts": [0, 19, 41],
                    "spans_ends": [10, 27, 48],
                    "labels": ["PER", "PER", "ORG"],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
                    "spans_starts": [],
                    "spans_ends": [],
                    "labels": [],
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
            },
        ]

        targets = [
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle."
                },
                "reference_fields": {
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
                "postprocessors": [
                    "processors.load_json",
                    "processors.dict_of_lists_to_value_key_pairs",
                ],
            },
            {
                "input_fields": {
                    "text": "John,: Doe is from New York and works at Goo:gle.",
                },
                "reference_fields": {
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
                "postprocessors": [
                    "processors.load_json",
                    "processors.dict_of_lists_to_value_key_pairs",
                ],
            },
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template(self):
        enumerators = ["capitals"]  # , "lowercase", "numbers", "roman"]
        firsts = ["A", "a", "1", "I"]
        seconds = ["B", "b", "2", "II"]
        for enumerator, first, second in zip(enumerators, firsts, seconds):
            template = MultipleChoiceTemplate(
                input_format="Text: {text}, Choices: {choices}.", enumerator=enumerator
            )

            choices = ["True", "False"]
            inputs = [
                {
                    "input_fields": {"choices": choices, "text": "example A"},
                    "reference_fields": {"choices": choices, "label": 0},
                },
                {
                    "input_fields": {"choices": choices, "text": "example A"},
                    "reference_fields": {"choices": choices, "label": "False"},
                },
                {
                    "input_fields": {"choices": ["True", "small"], "text": "example A"},
                    "reference_fields": {
                        "choices": ["True", "small"],
                        "label": "small",
                    },
                },
            ]

            targets = [
                {
                    "input_fields": {
                        "choices": choices,
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": choices,
                        "label": 0,
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {
                        "choices": choices,
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": choices,
                        "label": "False",
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. False.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {
                        "choices": ["True", "small"],
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": ["True", "small"],
                        "label": "small",
                    },
                    "source": f"Text: example A, Choices: {first}. True, {second}. small.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
            ]

            check_operator(template, inputs, targets, tester=self)

        # check error and more options, to code cover additional lines
        template = MultipleChoiceTemplate(
            input_format="Text: {no_text}, Choices: {no_choices}.",
            postprocessors=["post1", "post2"],
        )

        check_operator_exception(
            template,
            inputs,
            [
                "Available input fields are [numerals, choices, text] but MultipleChoiceTemplate.input_format format requires a different ones: 'Text: {no_text}, Choices: {no_choices}.'"
            ],
            tester=self,
        )

        # with self.assertRaises(ValueError) as ve:
        #     check_operator(template, inputs, targets, tester=self)
        # self.assertIn(
        #     "Available input fields are [numerals, choices, text] but MultipleChoiceTemplate.input_format format requires a different ones: 'Text: {no_text}, Choices: {no_choices}.'",
        #     str(ve.exception),
        # )

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
                    "input_fields": {"choices": ["True", "False"], "text": "example A"},
                    "reference_fields": {"choices": ["True", "False"], "label": 0},
                },
                {
                    "input_fields": {"choices": ["True", "False"], "text": "example A"},
                    "reference_fields": {
                        "choices": ["True", "False"],
                        "label": "False",
                    },
                },
                {
                    "input_fields": {"choices": ["True", temp], "text": "example A"},
                    "reference_fields": {"choices": ["True", temp], "label": temp},
                },
            ]

            targets = [
                {
                    "input_fields": {
                        "choices": ["False", "True"],
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": ["False", "True"],
                        "label": 1,
                    },
                    "source": f"Text: example A, Choices: {first}. False, {second}. True.",
                    "target": f"{second}",
                    "references": [f"{second}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {
                        "choices": ["False", "True"],
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": ["False", "True"],
                        "label": 0,
                    },
                    "source": f"Text: example A, Choices: {first}. False, {second}. True.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
                {
                    "input_fields": {
                        "choices": [temp, "True"],
                        "text": "example A",
                        "options": [f"{first}", f"{second}"],
                    },
                    "reference_fields": {
                        "choices": [temp, "True"],
                        "label": 0,
                    },
                    "source": f"Text: example A, Choices: {first}. {temp}, {second}. True.",
                    "target": f"{first}",
                    "references": [f"{first}"],
                    "instruction": "",
                    "target_prefix": "",
                    "postprocessors": ["processors.to_string_stripped"],
                },
            ]

            check_operator(template, inputs, targets, tester=self)

        # check error and more options, to code cover additional lines
        template = MultipleChoiceTemplate(
            input_format="Text: {no_text}, Choices: {no_choices}.",
            postprocessors=["post1", "post2"],
        )

        check_operator_exception(
            template,
            inputs,
            [
                "Available input fields are [numerals, choices, text] but MultipleChoiceTemplate.input_format format requires a different ones: 'Text: {no_text}, Choices: {no_choices}.'"
            ],
            tester=self,
        )

    def test_multiple_choice_template_with_shuffle_choices_seed(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            shuffle_choices=True,
            shuffle_choices_seed=12345,  # fixed seed
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["Alpha", "Beta", "Gamma"],
                    "text": "example seed",
                },
                "reference_fields": {
                    "choices": ["Alpha", "Beta", "Gamma"],
                    "label": 2,  # "Gamma" is index 2
                },
            }
        ]

        # Because we have a fixed seed, we expect a certain deterministic shuffle.
        # Let's assume that the random shuffle with seed=12345 will order the choices
        # as ["Beta", "Gamma", "Alpha"].
        # The enumerator is "capitals" => ["A", "B", "C"]
        targets = [
            {
                "input_fields": {
                    "choices": ["Gamma", "Beta", "Alpha"],  # after shuffling
                    "text": "example seed",
                    "options": ["A", "B", "C"],  # enumerator placeholders
                },
                "reference_fields": {
                    "choices": ["Gamma", "Beta", "Alpha"],
                    "label": 0,  # "Gamma" is now index 1
                },
                # The source with source_choice_format = "{choice_numeral}. {choice_text}"
                # => "A. Beta, B. Gamma, C. Alpha"
                "source": "Text: example seed, Choices: A. Gamma, B. Beta, C. Alpha.",
                "target": "A",  # Because original label=2 => "Gamma" => new index=1 => enumerator "B"
                "references": ["A"],  # For final references
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_sort_choices_by_length(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            sort_choices_by_length=True,
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["Large", "No", "Yes", "Tiny"],
                    "text": "example length sort",
                },
                "reference_fields": {
                    "choices": ["Large", "No", "Yes", "Tiny"],
                    "label": 0,  # "Large" is index 0
                },
            }
        ]

        # After sorting by length: ["No", "Yes", "Tiny", "Large"]
        # "Large" was originally label => now index=3 => enumerator "D"
        targets = [
            {
                "input_fields": {
                    "choices": ["No", "Yes", "Tiny", "Large"],
                    "text": "example length sort",
                    "options": ["A", "B", "C", "D"],  # enumerator placeholders
                },
                "reference_fields": {
                    "choices": ["No", "Yes", "Tiny", "Large"],
                    "label": 3,  # "Large" is now index 3
                },
                "source": "Text: example length sort, Choices: A. No, B. Yes, C. Tiny, D. Large.",
                "target": "D",  # enumerator => label=3 => "D"
                "references": ["D"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_sort_choices_alphabetically(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            sort_choices_alphabetically=True,
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["Delta", "Alpha", "Charlie", "Bravo"],
                    "text": "example alphabetical sort",
                },
                "reference_fields": {
                    "choices": ["Delta", "Alpha", "Charlie", "Bravo"],
                    "label": "Charlie",
                },
            }
        ]

        # Alphabetically sorted: ["Alpha", "Bravo", "Charlie", "Delta"]
        # "Charlie" => new index=2 => enumerator "C"
        targets = [
            {
                "input_fields": {
                    "choices": ["Alpha", "Bravo", "Charlie", "Delta"],
                    "text": "example alphabetical sort",
                    "options": ["A", "B", "C", "D"],
                },
                "reference_fields": {
                    "choices": ["Alpha", "Bravo", "Charlie", "Delta"],
                    "label": 2,  # "Charlie" is now index 2
                },
                "source": "Text: example alphabetical sort, Choices: A. Alpha, B. Bravo, C. Charlie, D. Delta.",
                "target": "C",
                "references": ["C"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_reverse_choices(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            reverse_choices=True,
            # No sorting flags => just reversing the original list
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["One", "Two", "Three"],
                    "text": "example reverse",
                },
                "reference_fields": {
                    "choices": ["One", "Two", "Three"],
                    "label": 1,  # "Two" is index 1
                },
            }
        ]

        # Original: ["One", "Two", "Three"]
        # Reversed: ["Three", "Two", "One"]
        # "Two" => new index=1 => enumerator "B"
        targets = [
            {
                "input_fields": {
                    "choices": ["Three", "Two", "One"],
                    "text": "example reverse",
                    "options": ["A", "B", "C"],  # enumerator
                },
                "reference_fields": {
                    "choices": ["Three", "Two", "One"],
                    "label": 1,  # "Two" is now index 1
                },
                "source": "Text: example reverse, Choices: A. Three, B. Two, C. One.",
                "target": "B",
                "references": ["B"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_place_correct_choice_position_first(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            place_correct_choice_position=0,
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["One", "Two", "Three"],
                    "text": "example place first",
                },
                "reference_fields": {
                    "choices": ["One", "Two", "Three"],
                    "label": 1,  # "Two" is index=1 in the original list
                },
            }
        ]

        # After placing correct choice ("Two") at index 0:
        # => ["Two", "One", "Three"]
        # => new label index=0 => enumerator => "A"
        targets = [
            {
                "input_fields": {
                    "choices": ["Two", "One", "Three"],
                    "text": "example place first",
                    "options": ["A", "B", "C"],  # enumerator placeholders
                },
                "reference_fields": {
                    "choices": ["Two", "One", "Three"],
                    "label": 0,  # "Two" is now index 0
                },
                "source": "Text: example place first, Choices: A. Two, B. One, C. Three.",
                "target": "A",  # label index=0 => "A"
                "references": ["A"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_place_correct_choice_position_last(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            place_correct_choice_position=2,  # for a 3-element list => "last" index is 2
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["One", "Two", "Three"],
                    "text": "example place last",
                },
                "reference_fields": {
                    "choices": ["One", "Two", "Three"],
                    "label": 0,  # "One" is index=0
                },
            }
        ]

        # After placing correct choice ("One") at index 2 (the last):
        # => ["Two", "Three", "One"]
        # => "One" is new index=2 => enumerator => "C"
        targets = [
            {
                "input_fields": {
                    "choices": ["Two", "Three", "One"],
                    "text": "example place last",
                    "options": ["A", "B", "C"],
                },
                "reference_fields": {
                    "choices": ["Two", "Three", "One"],
                    "label": 2,  # "One" is now index 2
                },
                "source": "Text: example place last, Choices: A. Two, B. Three, C. One.",
                "target": "C",
                "references": ["C"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_with_negative_index(self):
        template = MultipleChoiceTemplate(
            input_format="Text: {text}, Choices: {choices}.",
            enumerator="capitals",
            place_correct_choice_position=-1,  # Negative index to place correct choice last
        )

        inputs = [
            {
                "input_fields": {
                    "choices": ["Option A", "Option B", "Option C", "Option D"],
                    "text": "example negative index",
                },
                "reference_fields": {
                    "choices": ["Option A", "Option B", "Option C", "Option D"],
                    "label": 1,  # "Option B" is the correct choice at index 1
                },
            }
        ]

        # After placing correct choice ("Option B") at the last index:
        # => ["Option A", "Option C", "Option D", "Option B"]
        # => "Option B" is now at index 3 => enumerator => "D"
        targets = [
            {
                "input_fields": {
                    "choices": ["Option A", "Option C", "Option D", "Option B"],
                    "text": "example negative index",
                    "options": ["A", "B", "C", "D"],
                },
                "reference_fields": {
                    "choices": ["Option A", "Option C", "Option D", "Option B"],
                    "label": 3,  # "Option B" is now at index 3
                },
                "source": "Text: example negative index, Choices: A. Option A, B. Option C, C. Option D, D. Option B.",
                "target": "D",
                "references": ["D"],
                "instruction": "",
                "target_prefix": "",
                "postprocessors": ["processors.to_string_stripped"],
            }
        ]

        check_operator(template, inputs, targets, tester=self)

    def test_multiple_choice_template_verify_flags(self):
        with self.subTest("shuffle_choices + sort_choices_by_length"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    shuffle_choices=True,
                    sort_choices_by_length=True,
                )
                template.verify()
            self.assertIn("You cannot combine shuffle_choices", str(ve.exception))

        with self.subTest("shuffle_choices + sort_choices_alphabetically"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    shuffle_choices=True,
                    sort_choices_alphabetically=True,
                )
                template.verify()
            self.assertIn("You cannot combine shuffle_choices", str(ve.exception))

        with self.subTest("shuffle_choices + reverse_choices"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    shuffle_choices=True,
                    reverse_choices=True,
                )
                template.verify()
            self.assertIn("You cannot combine shuffle_choices", str(ve.exception))

        with self.subTest("sort_choices_by_length + sort_choices_alphabetically"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    sort_choices_by_length=True,
                    sort_choices_alphabetically=True,
                )
                template.verify()
            self.assertIn(
                "You cannot combine both sort_choices_by_length and sort_choices_alphabetically",
                str(ve.exception),
            )

        with self.subTest("shuffle_choices + place_correct_choice_position"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    shuffle_choices=True,
                    place_correct_choice_position=0,
                )
                template.verify()
            self.assertIn("You cannot combine shuffle_choices", str(ve.exception))

        with self.subTest("sort_choices_by_length + place_correct_choice_position"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    sort_choices_by_length=True,
                    place_correct_choice_position=0,
                )
                template.verify()
            self.assertIn(
                "You cannot combine place_correct_choice_position with sorting or reversing flags",
                str(ve.exception),
            )

        with self.subTest(
            "sort_choices_alphabetically + place_correct_choice_position"
        ):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    sort_choices_alphabetically=True,
                    place_correct_choice_position=0,
                )
                template.verify()
            self.assertIn(
                "You cannot combine place_correct_choice_position with sorting or reversing flags",
                str(ve.exception),
            )

        with self.subTest("reverse_choices + place_correct_choice_position"):
            with self.assertRaises(UnitxtError) as ve:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    reverse_choices=True,
                    place_correct_choice_position=0,
                )
                template.verify()
            self.assertIn(
                "You cannot combine place_correct_choice_position with sorting or reversing flags",
                str(ve.exception),
            )

        # Example of a "safe" scenario: place_correct_choice_position alone
        with self.subTest("place_correct_choice_position alone (should not raise)"):
            try:
                template = MultipleChoiceTemplate(
                    input_format="Text: {text}, Choices: {choices}.",
                    enumerator="capitals",
                    place_correct_choice_position=1,
                )
                template.verify()
            except UnitxtError as e:
                self.fail(f"Should not have raised an error, but got: {e}")

    def test_key_val_template_simple(self):
        template = KeyValTemplate()
        instance = {
            "input_fields": {"hello": "world", "str_list": ["djjd", "djjd"]},
            "reference_fields": {"label": "negative"},
        }
        result = template.process_instance(instance)

        self.assertEqual(result["target"], "negative")
        self.assertEqual(result["source"], "hello: world, str_list: djjd, djjd")

    def test_key_val_template_int_list(self):
        template = KeyValTemplate()
        instance = {
            "input_fields": {"hello": "world", "int_list": [0, 1]},
            "reference_fields": {"label": "negative"},
        }
        result = template.process_instance(instance)

        self.assertEqual(result["target"], "negative")
        self.assertEqual(result["source"], "hello: world, int_list: 0, 1")

    def test_render_template(self):
        instance = {
            "input_fields": {"text": "was so bad"},
            "reference_fields": {"label": "negative"},
        }
        template = InputOutputTemplate(
            input_format='This is my sentence: "{text}"', output_format="{label}"
        )

        result = template.process(instance)
        target = {
            "input_fields": {"text": "was so bad"},
            "reference_fields": {"label": "negative"},
            "source": 'This is my sentence: "was so bad"',
            "target": "negative",
            "references": ["negative"],
            "instruction": "",
            "target_prefix": "",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertDictEqual(result, target)

    def test_render_multi_reference_template(self):
        template = MultiReferenceTemplate(
            input_format="This is my sentence: {text}", references_field="answer"
        )
        instance = {
            "input_fields": {"text": "who was he?"},
            "reference_fields": {"answer": ["Dan", "Yossi"]},
        }

        result = template.process(instance)
        target = {
            "input_fields": {"text": "who was he?"},
            "reference_fields": {"answer": ["Dan", "Yossi"]},
            "source": "This is my sentence: who was he?",
            "target": "Dan",
            "references": ["Dan", "Yossi"],
            "instruction": "",
            "target_prefix": "",
            "postprocessors": ["processors.to_string_stripped"],
        }
        self.assertDictEqual(result, target)
