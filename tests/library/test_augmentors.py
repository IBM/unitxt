from unitxt.augmentors import AugmentPrefixSuffix, AugmentWhitespace, NullAugmentor
from unitxt.test_utils.operators import (
    apply_operator,
    check_operator_exception,
)

from tests.utils import UnitxtTestCase


class TestOperators(UnitxtTestCase):
    def test_augment_whitespace_task_input_with_error(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = AugmentWhitespace()
        operator.set_fields(["sentence"])
        with self.assertRaises(ValueError):
            apply_operator(operator, inputs)

    def test_type_dependent_augmentor_with_right_type(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = AugmentWhitespace(augmented_type=str)
        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        normalized_output_source = outputs[0]["input_fields"]["text"].split()
        normalized_input_source = text.split()
        self.assertEqual(normalized_output_source, normalized_input_source)
        self.assertNotEqual(text, outputs[0]["input_fields"]["text"])

    def test_type_dependent_augmentor_with_wrong_type(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = AugmentWhitespace(augmented_type=float)
        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        self.assertEqual(outputs[0]["input_fields"]["text"], text)

    def test_augment_whitespace_task_input(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = AugmentWhitespace()
        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        normalized_output_source = outputs[0]["input_fields"]["text"].split()
        normalized_input_source = text.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_null_task_input(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = NullAugmentor()
        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        normalized_output_source = outputs[0]["input_fields"]["text"]
        normalized_input_source = text
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_prefix_suffix_task_input(self):
        text = "\n She is riding a black horse  \t\t  "
        inputs = [{"input_fields": {"text": text}}]
        suffixes = ["Q", "R", "S", "T"]
        operator = AugmentPrefixSuffix(
            suffixes=suffixes,
            prefixes=None,
            remove_existing_whitespaces=True,
        )

        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        output0 = str(outputs[0]["input_fields"]["text"]).rstrip("".join(suffixes))
        assert (
            " \t\t " not in output0 and "\n" not in output0
        ), f"Leading and trailing whitespaces should have been removed, but still found in the output: {output0}"
        assert (
            output0 == text.strip()[: len(output0)]
        ), f"The prefix of {outputs[0]['input_fields']['text']!s} is not equal to the prefix of the stripped input: {text.strip()}"

    def test_augment_prefix_suffix_with_non_string_suffixes_error(self):
        prefixes = [10, 20, "O", "P"]
        with self.assertRaises(AssertionError) as ae:
            AugmentPrefixSuffix(prefixes=prefixes, suffixes=None)

        self.assertEqual(
            str(ae.exception),
            "Argument prefixes should be either None or a list of strings or a dictionary str->int. [10, 20, 'O', 'P'] is none of the above.",
        )

    def test_test_operator_unexpected_pass(self):
        text = "Should be ok"
        inputs = [{"input_fields": {"text": text}}]
        operator = AugmentWhitespace()
        operator.set_fields(["text"])
        exception_texts = [
            "Error processing instance '0' from stream 'test' in AugmentWhitespace due to the exception above.",
            "Error augmenting value 'None' from 'input_fields/text' in instance: {'input_fields': {'text': None}}",
        ]

        try:
            check_operator_exception(
                operator,
                inputs,
                exception_texts=exception_texts,
                tester=self,
            )
        except Exception as e:
            self.assertEqual(
                str(e),
                "Did not receive expected exception: Error processing instance '0' from stream 'test' in AugmentWhitespace due to the exception above.",
            )
