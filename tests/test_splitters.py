import unittest

from src.unitxt.splitters import DiverseLabelsSampler


class TestDiverseLabelsSampler(unittest.TestCase):
    """Tests for the DiverseLabelsSampler object."""

    @staticmethod
    def new_examplar(input_choices=["class_a", "class_b"], output_choices=["class_a"]):
        """return an examplar in a correct format."""
        return {
            "inputs": {"choices": input_choices},
            "outputs": {
                "choices": output_choices,
            },
        }

    def test_examplar_repr(self):
        sampler = DiverseLabelsSampler()
        expected_results = ["class_a"]
        result = sampler.examplar_repr(examplar=self.new_examplar())
        self.assertEqual(str(expected_results), result)

    def test_examplar_repr_with_string_for_input_choices(self):
        sampler = DiverseLabelsSampler()
        examplar_input_choices = "a string which is a wrong value"
        wrong_examplar = self.new_examplar(input_choices=examplar_input_choices)
        with self.assertRaises(ValueError) as cm:
            sampler.examplar_repr(examplar=wrong_examplar)
        self.assertEquals(
            f"Unexpected input choices value '{examplar_input_choices}'. Expected a list.",
            str(cm.exception),
        )

    def _test_examplar_repr_missing_field(self, missing_field):
        examplar = self.new_examplar()
        del examplar[missing_field]
        with self.assertRaises(ValueError) as cm:
            sampler = DiverseLabelsSampler()
            sampler.examplar_repr(examplar=examplar)
        self.assertEquals(
            f"'{missing_field}' field is missing from '{examplar}'.",
            str(cm.exception),
        )

    def test_examplar_repr_missing_fields(self):
        self._test_examplar_repr_missing_field(missing_field="inputs")
        self._test_examplar_repr_missing_field(missing_field="outputs")
