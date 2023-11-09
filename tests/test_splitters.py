import unittest

from unitxt.splitters import DiverseLabelsSampler


class TestDiverseLabelsSampler(unittest.TestCase):
    """
    Tests for the DiverseLabelsSampler object.
    """

    @staticmethod
    def get_correct_examplar():
        """
        return an examplar in a correct format.
        """
        return {
            "inputs": {"choices": ["class_a", "class_b"]},
            "outputs": {
                "choices": ["class_a"],
            },
        }

    def test_examplar_repr(self):
        sampler = DiverseLabelsSampler()
        expected_results = ["class_a"]
        result = sampler.examplar_repr(examplar=self.get_correct_examplar())
        self.assertEqual(str(expected_results), result)

    def _test_examplar_repr_missing_field(self, missing_field):
        examplar = self.get_correct_examplar()
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
