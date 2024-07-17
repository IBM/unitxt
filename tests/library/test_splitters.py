import copy

from unitxt.splitters import DiverseLabelsSampler

from tests.utils import UnitxtTestCase


class TestDiverseLabelsSampler(UnitxtTestCase):
    """Tests for the DiverseLabelsSampler object."""

    @staticmethod
    def new_exemplar(choices=None, labels=None, text=""):
        """Return an exemplar in a correct format."""
        if labels is None:
            labels = ["class_a"]
        if choices is None:
            choices = ["class_a", "class_b"]
        return {
            "input_fields": {"choices": choices, "text": text},
            "reference_fields": {
                "labels": labels,
            },
        }

    def test_sample(self):
        for i in range(3):
            num_samples = 3
            sampler = DiverseLabelsSampler(num_samples)
            choices = ["dog", "cat"]
            instances = [
                self.new_exemplar(choices, ["dog"], "Bark1"),
                self.new_exemplar(choices, ["dog"], "Bark2"),
                self.new_exemplar(choices, ["cat"], "Cat1"),
                self.new_exemplar(choices, ["dog"], "Bark3"),
                self.new_exemplar(choices, ["cow"], "Moo1"),
                self.new_exemplar(choices, ["duck"], "Quack"),
            ]
            result = sampler.sample(instances)

            from collections import Counter

            counts = Counter()
            for i in range(0, num_samples):
                counts[result[i]["reference_fields"]["labels"][0]] += 1
            self.assertEqual(counts["dog"], 1)
            self.assertEqual(counts["cat"], 1)
            self.assertEqual(len(counts.keys()), 3)

    def test_sample_no_empty_labels(self):
        for i in range(3):
            num_samples = 3
            sampler = DiverseLabelsSampler(num_samples, include_empty_label=False)
            choices = ["dog", "cat"]
            instances = [
                self.new_exemplar(choices, ["dog"], "Bark1"),
                self.new_exemplar(choices, ["dog"], "Bark2"),
                self.new_exemplar(choices, ["cat"], "Cat1"),
                self.new_exemplar(choices, ["dog"], "Bark3"),
                self.new_exemplar(choices, ["cow"], "Moo1"),
                self.new_exemplar(choices, ["duck"], "Quack"),
            ]
            result = sampler.sample(instances)

            from collections import Counter

            counts = Counter()
            for i in range(0, num_samples):
                counts[result[i]["reference_fields"]["labels"][0]] += 1
            self.assertEqual(set(counts.keys()), {"dog", "cat"})

    def test_sample_list(self):
        for _ in range(10):
            num_samples = 2
            sampler = DiverseLabelsSampler(num_samples)
            choices = ["cat"]
            instances = [
                self.new_exemplar(choices, ["dog", "cat"], "Bark1,Cat1"),
                self.new_exemplar(choices, ["cat"], "Cat2"),
                self.new_exemplar(choices, ["dog"], "Bark2"),
                self.new_exemplar(choices, ["duck"], "Quack"),
            ]
            result = sampler.sample(instances)
            from collections import Counter

            counts = Counter()
            for j in range(0, num_samples):
                counts[str(result[j]["reference_fields"]["labels"])] += 1
            self.assertTrue(
                counts["['dog', 'cat']"] == 1 or counts["['cat']"] == 1,
                f"unexpected counts: {counts}",
            )
            self.assertTrue(
                counts["['duck']"] == 1 or counts["['dog']"] == 1,
                f"unexpected counts: {counts}",
            )

    def test_exemplar_repr(self):
        sampler = DiverseLabelsSampler()
        expected_results = ["class_a"]
        result = sampler.exemplar_repr(exemplar=self.new_exemplar())
        self.assertEqual(str(expected_results), result)

    def test_exemplar_repr_with_string_for_input_choices(self):
        sampler = DiverseLabelsSampler()
        exemplar_input_choices = {"not": "good"}
        wrong_exemplar = self.new_exemplar(choices=exemplar_input_choices)
        with self.assertRaises(ValueError) as cm:
            sampler.exemplar_repr(exemplar=wrong_exemplar)
        self.assertEqual(
            f"Unexpected input choices value '{exemplar_input_choices}'. Expected a list or a string.",
            str(cm.exception),
        )

    def _test_exemplar_repr_missing_field(self, missing_field):
        exemplar = self.new_exemplar()
        del exemplar[missing_field]
        with self.assertRaises(ValueError) as cm:
            sampler = DiverseLabelsSampler()
            sampler.exemplar_repr(exemplar=exemplar)
        self.assertEqual(
            f"'{missing_field}' field is missing from '{exemplar}'.",
            str(cm.exception),
        )

    def test_exemplar_repr_missing_fields(self):
        self._test_exemplar_repr_missing_field(missing_field="input_fields")
        self._test_exemplar_repr_missing_field(missing_field="reference_fields")

    def test_filter_with_bad_input(self):
        sampler = DiverseLabelsSampler(3)
        choices = ["dog", "cat"]
        instances = [
            self.new_exemplar(choices, ["dog"], "Bark1"),
            self.new_exemplar(choices, ["dog"], "Bark2"),
            self.new_exemplar(choices, ["cat"], "Cat1"),
        ]
        instance = copy.deepcopy(instances[0])

        filtered_instances = sampler.filter_source_by_instance(instances, instance)
        self.assertEqual(len(filtered_instances), 2)

        del instance["input_fields"]
        with self.assertRaises(ValueError) as cm:
            sampler.filter_source_by_instance(instances, instance)
        self.assertEqual(
            f"'input_fields' field is missing from '{instance}'.",
            str(cm.exception),
        )
