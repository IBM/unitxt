import copy

from src.unitxt.splitters import DiverseLabelsSampler
from tests.utils import UnitxtTestCase


class TestDiverseLabelsSampler(UnitxtTestCase):
    """Tests for the DiverseLabelsSampler object."""

    @staticmethod
    def new_examplar(choices=None, labels=None, text=""):
        """Return an examplar in a correct format."""
        if labels is None:
            labels = ["class_a"]
        if choices is None:
            choices = ["class_a", "class_b"]
        return {
            "inputs": {"choices": choices, "text": text},
            "outputs": {
                "labels": labels,
            },
        }

    def test_sample(self):
        for i in range(3):
            num_samples = 3
            sampler = DiverseLabelsSampler(num_samples)
            choices = ["dog", "cat"]
            instances = [
                self.new_examplar(choices, ["dog"], "Bark1"),
                self.new_examplar(choices, ["dog"], "Bark2"),
                self.new_examplar(choices, ["cat"], "Cat1"),
                self.new_examplar(choices, ["dog"], "Bark3"),
                self.new_examplar(choices, ["cow"], "Moo1"),
                self.new_examplar(choices, ["duck"], "Quack"),
            ]
            result = sampler.sample(instances)

            from collections import Counter

            counts = Counter()
            for i in range(0, num_samples):
                counts[result[i]["outputs"]["labels"][0]] += 1
            self.assertEqual(counts["dog"], 1)
            self.assertEqual(counts["cat"], 1)
            self.assertEqual(len(counts.keys()), 3)

    def test_sample_list(self):
        for _ in range(10):
            num_samples = 2
            sampler = DiverseLabelsSampler(num_samples)
            choices = ["cat"]
            instances = [
                self.new_examplar(choices, ["dog", "cat"], "Bark1,Cat1"),
                self.new_examplar(choices, ["cat"], "Cat2"),
                self.new_examplar(choices, ["dog"], "Bark2"),
                self.new_examplar(choices, ["duck"], "Quack"),
            ]
            result = sampler.sample(instances)
            from collections import Counter

            counts = Counter()
            for j in range(0, num_samples):
                counts[str(result[j]["outputs"]["labels"])] += 1
            self.assertTrue(
                counts["['dog', 'cat']"] == 1 or counts["['cat']"] == 1,
                f"unexpected counts: {counts}",
            )
            self.assertTrue(
                counts["['duck']"] == 1 or counts["['dog']"] == 1,
                f"unexpected counts: {counts}",
            )

    def test_examplar_repr(self):
        sampler = DiverseLabelsSampler()
        expected_results = ["class_a"]
        result = sampler.examplar_repr(examplar=self.new_examplar())
        self.assertEqual(str(expected_results), result)

    def test_examplar_repr_with_string_for_input_choices(self):
        sampler = DiverseLabelsSampler()
        examplar_input_choices = "a string which is a wrong value"
        wrong_examplar = self.new_examplar(choices=examplar_input_choices)
        with self.assertRaises(ValueError) as cm:
            sampler.examplar_repr(examplar=wrong_examplar)
        self.assertEqual(
            f"Unexpected input choices value '{examplar_input_choices}'. Expected a list.",
            str(cm.exception),
        )

    def _test_examplar_repr_missing_field(self, missing_field):
        examplar = self.new_examplar()
        del examplar[missing_field]
        with self.assertRaises(ValueError) as cm:
            sampler = DiverseLabelsSampler()
            sampler.examplar_repr(examplar=examplar)
        self.assertEqual(
            f"'{missing_field}' field is missing from '{examplar}'.",
            str(cm.exception),
        )

    def test_examplar_repr_missing_fields(self):
        self._test_examplar_repr_missing_field(missing_field="inputs")
        self._test_examplar_repr_missing_field(missing_field="outputs")

    def test_filter_with_bad_input(self):
        sampler = DiverseLabelsSampler(3)
        choices = ["dog", "cat"]
        instances = [
            self.new_examplar(choices, ["dog"], "Bark1"),
            self.new_examplar(choices, ["dog"], "Bark2"),
            self.new_examplar(choices, ["cat"], "Cat1"),
        ]
        instance = copy.deepcopy(instances[0])

        filtered_instances = sampler.filter_source_by_instance(instances, instance)
        self.assertEqual(len(filtered_instances), 2)

        del instance["inputs"]
        with self.assertRaises(ValueError) as cm:
            sampler.filter_source_by_instance(instances, instance)
        self.assertEqual(
            f"'inputs' field is missing from '{instance}'.",
            str(cm.exception),
        )
