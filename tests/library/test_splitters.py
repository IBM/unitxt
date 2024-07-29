import copy

from unitxt.api import load_dataset
from unitxt.blocks import TaskCard
from unitxt.collections_operators import Wrap
from unitxt.loaders import LoadFromDictionary
from unitxt.splitters import CloseTextSampler, DiverseLabelsSampler, FixedIndicesSampler

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
            result = sampler.sample(
                instances,
                self.new_exemplar(choices, ["any"], "any"),
            )

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
            result = sampler.sample(
                instances,
                self.new_exemplar(choices, ["any"], "any"),
            )

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
            result = sampler.sample(
                instances, self.new_exemplar(choices, ["any"], "any")
            )
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


class TestCloseTextSampler(UnitxtTestCase):
    """Tests for the CloseTextSampler object."""

    @staticmethod
    def new_exemplar(question: str, answer: str):
        """Return an exemplar in a correct format."""
        return {
            "input_fields": {"question": question, "answer": answer},
        }

    def test_sample(self):
        instances = [
            self.new_exemplar("What is your name?", "John"),
            self.new_exemplar("In which country is Paris located?", "France"),
            self.new_exemplar("What's the time?", "22:00"),
            self.new_exemplar("What is your name, please?", "Mary"),
        ]

        num_samples = 2
        sampler = CloseTextSampler(num_samples, field="question")

        results = sampler.sample(
            instances, self.new_exemplar("What's your name?", "don't know")
        )
        self.assertEqual(results, [instances[0], instances[3]])

        results = sampler.sample(
            instances, self.new_exemplar("What is the time?", "don't know")
        )
        self.assertEqual(results, [instances[2], instances[0]])

        num_samples = 1
        sampler = CloseTextSampler(num_samples, field="answer")
        results = sampler.sample(
            instances, self.new_exemplar("Who do I love?", "Mary Lu")
        )
        self.assertEqual(results, [instances[3]])

    def test_filter_with_wrong_field(self):
        num_samples = 2
        sampler = CloseTextSampler(num_samples, field="wrong_field")
        instances = [
            self.new_exemplar("What is your name?", "John"),
        ]
        instance = self.new_exemplar("What's your name?", "don't know")
        with self.assertRaises(ValueError) as cm:
            sampler.sample(instances, instance)
        self.assertIn(
            'query "input_fields/wrong_field" did not match any item in dict',
            str(cm.exception),
        )

    def test_end2end(self):
        data = {
            "train": [
                {"question": "What is your name?", "answer": "John"},
                {"question": "In which country is Paris located?", "answer": "France"},
                {"question": "At what time do we they eat dinner?", "answer": "22:00"},
                {"question": "What's your name, please?", "answer": "Mary"},
                {"question": "Is this your car?", "answer": "yes"},
                {"question": "What is your name?", "answer": "Sunny"},
            ],
            "test": [
                {"question": "What's your name?", "answer": "John"},
            ],
        }

        card = TaskCard(
            loader=LoadFromDictionary(data=data),
            task="tasks.qa.open",
            preprocess_steps=[Wrap(field="answer", inside="list", to_field="answers")],
        )

        dataset = load_dataset(
            card=card,
            template="templates.qa.open.title",
            demos_pool_size=5,
            num_demos=2,
            sampler=CloseTextSampler(field="question"),
        )
        expected_output = """Answer the question.
Question:
What is your name?
Answer:
John

Question:
What's your name, please?
Answer:
Mary

Question:
What's your name?
Answer:
"""
        self.assertEqual(dataset["test"][0]["source"], expected_output)


class TestFixedIndicesSampler(UnitxtTestCase):
    """Tests for the FixedIndicesSampler  object."""

    @staticmethod
    def new_exemplar(question: str, answer: str):
        """Return an exemplar in a correct format."""
        return {
            "input_fields": {"question": question, "answer": answer},
        }

    def test_sample(self):
        instances = [
            self.new_exemplar("What is your name?", "John"),
            self.new_exemplar("In which country is Paris located?", "France"),
            self.new_exemplar("What's the time?", "22:00"),
            self.new_exemplar("What is your name, please?", "Mary"),
        ]
        instance = self.new_exemplar("What's your name?", "don't know")
        sampler = FixedIndicesSampler(indices=[2, 0])

        results = sampler.sample(instances, instance)
        self.assertEqual(results, [instances[2], instances[0]])

    def test_out_of_bound_sample(self):
        instances = [
            self.new_exemplar("What is your name?", "John"),
            self.new_exemplar("In which country is Paris located?", "France"),
        ]

        instance = self.new_exemplar("What's your name?", "don't know")
        sampler = FixedIndicesSampler(indices=[2])
        with self.assertRaises(ValueError) as cm:
            sampler.sample(instances, instance)
        self.assertIn(
            "FixedIndicesSampler 'indices' field contains index (2) which is out of bounds of the instance pool ( of size 2)",
            str(cm.exception),
        )
