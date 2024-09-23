from unitxt.benchmark import Benchmark
from unitxt.standard import StandardRecipe

from tests.utils import UnitxtTestCase


class TestBenchmark(UnitxtTestCase):
    def test_benchmark(self):
        benchmark = Benchmark(
            format="formats.user_agent",
            max_samples_per_subset=2,
            loader_limit=300,
            subsets={
                "cola": StandardRecipe(
                    card="cards.cola",
                    template="templates.classification.multi_class.instruction",
                ),
                "wnli": StandardRecipe(
                    card="cards.wnli",
                    template="templates.classification.multi_class.relation.default",
                ),
            },
        )

        test_dataset = list(benchmark()["test"])

        self.assertListEqual(
            test_dataset,
            [
                {
                    "metrics": ["metrics.matthews_correlation"],
                    "data_classification_policy": ["public"],
                    "media": {"images": [], "audios": []},
                    "postprocessors": [
                        "processors.take_first_non_empty_line",
                        "processors.lower_case_till_punc",
                    ],
                    "target": "acceptable",
                    "references": ["acceptable"],
                    "source": "Classify the grammatical acceptability of the following text to one of these options: unacceptable, acceptable.\n\nUser:text: The sailors rode the breeze clear of the rocks.\nAgent:The grammatical acceptability is ",
                    "task_data": '{"text": "The sailors rode the breeze clear of the rocks.", "text_type": "text", "classes": ["unacceptable", "acceptable"], "type_of_class": "grammatical acceptability", "metadata": {"data_classification_policy": ["public"], "num_demos": 0, "template": "templates.classification.multi_class.instruction"}, "label": "acceptable"}',
                    "groups": [],
                    "subset": ["cola"],
                },
                {
                    "metrics": ["metrics.matthews_correlation"],
                    "data_classification_policy": ["public"],
                    "media": {"images": [], "audios": []},
                    "postprocessors": [
                        "processors.take_first_non_empty_line",
                        "processors.lower_case_till_punc",
                    ],
                    "target": "acceptable",
                    "references": ["acceptable"],
                    "source": "Classify the grammatical acceptability of the following text to one of these options: unacceptable, acceptable.\n\nUser:text: The weights made the rope stretch over the pulley.\nAgent:The grammatical acceptability is ",
                    "task_data": '{"text": "The weights made the rope stretch over the pulley.", "text_type": "text", "classes": ["unacceptable", "acceptable"], "type_of_class": "grammatical acceptability", "metadata": {"data_classification_policy": ["public"], "num_demos": 0, "template": "templates.classification.multi_class.instruction"}, "label": "acceptable"}',
                    "groups": [],
                    "subset": ["cola"],
                },
                {
                    "metrics": [
                        "metrics.f1_micro",
                        "metrics.accuracy",
                        "metrics.f1_macro",
                    ],
                    "data_classification_policy": ["public"],
                    "media": {"images": [], "audios": []},
                    "postprocessors": [
                        "processors.take_first_non_empty_line",
                        "processors.lower_case_till_punc",
                    ],
                    "target": "entailment",
                    "references": ["entailment"],
                    "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\n\nUser:premise: The drain is clogged with hair. It has to be cleaned.\nhypothesis: The hair has to be cleaned.\nAgent:The entailment class is ",
                    "task_data": '{"text_a": "The drain is clogged with hair. It has to be cleaned.", "text_a_type": "premise", "text_b": "The hair has to be cleaned.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"], "num_demos": 0, "template": "templates.classification.multi_class.relation.default"}, "label": "entailment"}',
                    "groups": [],
                    "subset": ["wnli"],
                },
                {
                    "metrics": [
                        "metrics.f1_micro",
                        "metrics.accuracy",
                        "metrics.f1_macro",
                    ],
                    "data_classification_policy": ["public"],
                    "media": {"images": [], "audios": []},
                    "postprocessors": [
                        "processors.take_first_non_empty_line",
                        "processors.lower_case_till_punc",
                    ],
                    "target": "not entailment",
                    "references": ["not entailment"],
                    "source": "Given a premise and hypothesis classify the entailment of the hypothesis to one of entailment, not entailment.\n\nUser:premise: Jane knocked on Susan's door but she did not answer.\nhypothesis: Susan did not answer.\nAgent:The entailment class is ",
                    "task_data": '{"text_a": "Jane knocked on Susan\'s door but she did not answer.", "text_a_type": "premise", "text_b": "Susan did not answer.", "text_b_type": "hypothesis", "classes": ["entailment", "not entailment"], "type_of_relation": "entailment", "metadata": {"data_classification_policy": ["public"], "num_demos": 0, "template": "templates.classification.multi_class.relation.default"}, "label": "not entailment"}',
                    "groups": [],
                    "subset": ["wnli"],
                },
            ],
        )
