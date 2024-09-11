from collections import Counter

from unitxt.augmentors import (
    AugmentPrefixSuffix,
    AugmentWhitespace,
    ModelInputAugmentor,
    TaskInputsAugmentor,
)
from unitxt.test_utils.operators import (
    apply_operator,
    check_operator_exception,
)

from tests.utils import UnitxtTestCase


class TestOperators(UnitxtTestCase):
    def test_augment_whitespace_model_input(self):
        source = "The dog ate my cat"
        inputs = [{"source": source}]

        operator = ModelInputAugmentor(operator=AugmentWhitespace())
        outputs = apply_operator(operator, inputs)
        assert (
            outputs[0]["source"] != source
        ), f"Source of f{outputs} is equal to f{source} and was not augmented"
        normalized_output_source = outputs[0]["source"].split()
        normalized_input_source = source.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_whitespace_task_input_with_error(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = TaskInputsAugmentor(operator=AugmentWhitespace())
        operator.set_fields(["sentence"])
        with self.assertRaises(ValueError):
            apply_operator(operator, inputs)

    def test_augment_whitespace_task_input(self):
        text = "The dog ate my cat"
        inputs = [{"input_fields": {"text": text}}]
        operator = TaskInputsAugmentor(operator=AugmentWhitespace())
        operator.set_fields(["text"])
        outputs = apply_operator(operator, inputs)
        normalized_output_source = outputs[0]["input_fields"]["text"].split()
        normalized_input_source = text.split()
        assert (
            normalized_output_source == normalized_input_source
        ), f"{normalized_output_source} is not equal to f{normalized_input_source}"

    def test_augment_whitespace_with_none_text_error(self):
        text = None
        inputs = [{"input_fields": {"text": text}}]
        operator = TaskInputsAugmentor(operator=AugmentWhitespace())
        operator.set_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in TaskInputsAugmentor due to: Failed to process 'input_fields/text' from {'input_fields': {'text': None}} due to : expected string or bytes-like object"
        check_operator_exception(
            operator,
            inputs,
            tester=self,
            exception_text=exception_text,
        )

    def test_augment_prefix_suffix_model_input(self):
        source = "\n He is riding a black horse\t\t  "
        inputs = [{"source": source}]
        prefixes = [
            "M",
            "N",
            "O",
            "P",
        ]  # all distinct from source, to ease verification
        suffixes = [
            "Q",
            "R",
            "S",
            "T",
        ]  # all distinct from source, to ease verification

        operator = ModelInputAugmentor(
            operator=AugmentPrefixSuffix(suffixes=suffixes, prefixes=prefixes)
        )
        outputs = apply_operator(operator, inputs)
        assert (
            outputs[0]["source"] != source
        ), f"Output remains equal to source, {source}, and was not augmented"
        output0 = str(outputs[0]["source"]).strip("".join(prefixes + suffixes))
        assert (
            output0 == source
        ), f"The inner part of the output, {outputs[0]['source']}, is not equal to the input {source}"
        assert (
            "\t\t " in output0
        ), f"Trailing whitespaces wrongly removed, yielding {output0}, although 'remove_existing_whitespaces' is False,"
        # weighted suffixes
        suffixes_dict = {"Q": 2, "R": 2, "S": 2, "T": 10}
        operator = ModelInputAugmentor(
            operator=AugmentPrefixSuffix(
                suffixes=suffixes_dict,
                suffix_len=8,
                prefixes=None,
            )
        )
        outputs = apply_operator(operator, [({"source": str(i)}) for i in range(500)])
        assert (
            len(outputs) == 500
        ), f"outputs length {len(outputs)} is different from inputs length, which is 500."
        actual_suffixes = [output["source"][-2:] for output in outputs]
        counter = Counter(actual_suffixes)
        assert (
            counter["TT"] > counter["SS"]
        ), f'In a population of size 500 , suffix "TT" ({counter["TT"]}) is expected to be more frequent than "SS" {counter["SS"]}'

    def test_augment_prefix_suffix_task_input(self):
        text = "\n She is riding a black horse  \t\t  "
        inputs = [{"input_fields": {"text": text}}]
        suffixes = ["Q", "R", "S", "T"]
        operator = TaskInputsAugmentor(
            operator=AugmentPrefixSuffix(
                suffixes=suffixes,
                prefixes=None,
                remove_existing_whitespaces=True,
            )
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
            TaskInputsAugmentor(
                operator=AugmentPrefixSuffix(prefixes=prefixes, suffixes=None)
            )
        self.assertEqual(
            str(ae.exception),
            "Argument prefixes should be either None or a list of strings or a dictionary str->int. [10, 20, 'O', 'P'] is none of the above.",
        )

    def test_augment_prefix_suffix_with_none_input_error(self):
        text = None
        inputs = [{"input_fields": {"text": text}}]
        suffixes = ["Q", "R", "S", "T"]
        operator = TaskInputsAugmentor(
            operator=AugmentPrefixSuffix(suffixes=suffixes, prefixes=None)
        )
        operator.set_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in TaskInputsAugmentor due to: Failed to process 'input_fields/text' from {'input_fields': {'text': None}} due to : input value should not be None"
        check_operator_exception(
            operator,
            inputs,
            tester=self,
            exception_text=exception_text,
        )

    def test_test_operator_without_tester_param(self):
        text = None
        inputs = [{"input_fields": {"text": text}}]
        operator = TaskInputsAugmentor(operator=AugmentWhitespace())
        operator.set_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in TaskInputsAugmentor due to: Failed to process 'input_fields/text' from {'input_fields': {'text': None}} due to : expected string or bytes-like object"

        check_operator_exception(
            operator,
            inputs,
            exception_text=exception_text,
            tester=self,
        )

    def test_test_operator_unexpected_pass(self):
        text = "Should be ok"
        inputs = [{"input_fields": {"text": text}}]
        operator = TaskInputsAugmentor(operator=AugmentWhitespace())
        operator.set_fields(["text"])
        exception_text = "Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'input_fields/text' in instance: {'input_fields': {'text': None}}"

        try:
            check_operator_exception(
                operator,
                inputs,
                exception_text=exception_text,
                tester=self,
            )
        except Exception as e:
            self.assertEqual(
                str(e),
                "Did not receive expected exception Error processing instance '0' from stream 'test' in AugmentWhitespace due to: Error augmenting value 'None' from 'input_fields/text' in instance: {'input_fields': {'text': None}}",
            )
