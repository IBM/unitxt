from unitxt.api import evaluate
from unitxt.fusion import FixedFusion, WeightedFusion
from unitxt.operators import IterableSource
from unitxt.standard import DatasetRecipe
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase, fillna, round_values


class TestFusion(UnitxtTestCase):
    def test_unbounded_fixed_fusion(self):
        operator = FixedFusion(
            subsets={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
        )

        targets = [
            {"x": "x1", "subset": ["origin1"]},
            {"x": "x2", "subset": ["origin1"]},
            {"x": "x3", "subset": ["origin1"]},
            {"x": "y1", "subset": ["origin2"]},
            {"x": "y2", "subset": ["origin2"]},
            {"x": "y3", "subset": ["origin2"]},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

        # test the saving of past groups
        origin3 = [{"x": "z1"}, {"x": "z2"}, {"x": "z3"}]
        operator = FixedFusion(
            subsets={
                "targets1": IterableSource({"test": targets}),
                "origin3": IterableSource({"test": origin3}),
            },
        )

        targets2 = [
            {"x": "x1", "subset": ["targets1", "origin1"]},
            {"x": "x2", "subset": ["targets1", "origin1"]},
            {"x": "x3", "subset": ["targets1", "origin1"]},
            {"x": "y1", "subset": ["targets1", "origin2"]},
            {"x": "y2", "subset": ["targets1", "origin2"]},
            {"x": "y3", "subset": ["targets1", "origin2"]},
            {"x": "z1", "subset": ["origin3"]},
            {"x": "z2", "subset": ["origin3"]},
            {"x": "z3", "subset": ["origin3"]},
        ]
        check_operator(operator, inputs=None, targets=targets2, tester=self)

    def compare_stream(self, stream, expected_stream):
        self.assertEqual(len(stream), len(expected_stream))
        for input_dict, output_dict in zip(stream, expected_stream):
            self.assertDictEqual(input_dict, output_dict)

    def test_nonoverlapping_splits_fusion(self):
        operator = FixedFusion(
            include_splits=["train", "test"],
            subsets={
                "origin_train": IterableSource(
                    {"train": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin_test": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
        )

        output_multi_stream = operator()
        self.assertListEqual(sorted(output_multi_stream.keys()), ["test", "train"])
        self.compare_stream(
            list(output_multi_stream["train"]),
            [
                {"x": "x1", "subset": ["origin_train"]},
                {"x": "x2", "subset": ["origin_train"]},
                {"x": "x3", "subset": ["origin_train"]},
            ],
        )
        self.compare_stream(
            list(output_multi_stream["test"]),
            [
                {"x": "y1", "subset": ["origin_test"]},
                {"x": "y2", "subset": ["origin_test"]},
                {"x": "y3", "subset": ["origin_test"]},
            ],
        )

    def test_bounded_fixed_fusion(self):
        fixed_fusion = FixedFusion(
            subsets={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            max_instances_per_subset=2,
        )
        targets = [
            {"x": "x1", "subset": ["origin1"]},
            {"x": "x2", "subset": ["origin1"]},
            {"x": "y1", "subset": ["origin2"]},
            {"x": "y2", "subset": ["origin2"]},
        ]
        outputs = fixed_fusion()
        self.compare_stream(targets, list(outputs["test"]))

    def test_over_bounded_fixed_fusion(self):
        operator = FixedFusion(
            subsets={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            max_instances_per_subset=10,
        )

        targets = [
            {"x": "x1", "subset": ["origin1"]},
            {"x": "x2", "subset": ["origin1"]},
            {"x": "x3", "subset": ["origin1"]},
            {"x": "y1", "subset": ["origin2"]},
            {"x": "y2", "subset": ["origin2"]},
            {"x": "y3", "subset": ["origin2"]},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

    def test_unbounded_weighted_fusion(self):
        operator = WeightedFusion(
            subsets={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            weights={"origin1": 1, "origin2": 1},
        )

        targets = [
            {"x": "x1", "subset": ["origin1"]},
            {"x": "x2", "subset": ["origin1"]},
            {"x": "x3", "subset": ["origin1"]},
            {"x": "y1", "subset": ["origin2"]},
            {"x": "y2", "subset": ["origin2"]},
            {"x": "y3", "subset": ["origin2"]},
        ]

        check_operator(
            operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        )

    def test_over_bounded_weighted_fusion(self):
        operator = WeightedFusion(
            subsets={
                "origin1": IterableSource(
                    {
                        "test": [
                            {"x": "x1"},
                            {"x": "x2"},
                            {"x": "x3"},
                            {"x": "x4"},
                            {"x": "x5"},
                        ],
                        "train": [
                            {"a": "x1"},
                            {"a": "x2"},
                            {"a": "x3"},
                            {"a": "x4"},
                            {"a": "x5"},
                        ],
                    }
                ),
                "origin2": IterableSource(
                    {
                        "test": [
                            {"x": "y1"},
                            {"x": "y2"},
                            {"x": "y3"},
                            {"x": "y4"},
                            {"x": "y5"},
                        ],
                        "train": [
                            {"b": "y1"},
                            {"b": "y2"},
                            {"b": "y3"},
                            {"b": "y4"},
                            {"b": "y5"},
                        ],
                    }
                ),
            },
            weights={"origin1": 1, "origin2": 2},
            max_total_samples=3,
        )

        res = operator()
        targets = {
            "test": [
                {"x": "y1", "subset": ["origin2"]},
                {"x": "x1", "subset": ["origin1"]},
                {"x": "y2", "subset": ["origin2"]},
            ],
            "train": [
                {"a": "x1", "subset": ["origin1"]},
                {"b": "y1", "subset": ["origin2"]},
                {"b": "y2", "subset": ["origin2"]},
            ],
        }
        for key in ["train", "test"]:
            self.compare_stream(targets[key], list(res[key]))

        operator = WeightedFusion(
            subsets={
                "origin1": IterableSource(
                    {
                        "test": [
                            {"x": "x1"},
                            {"x": "x2"},
                            {"x": "x3"},
                            {"x": "x4"},
                            {"x": "x5"},
                        ],
                        "train": [
                            {"a": "x1"},
                            {"a": "x2"},
                            {"a": "x3"},
                            {"a": "x4"},
                            {"a": "x5"},
                        ],
                    }
                ),
                "origin2": IterableSource(
                    {
                        "test": [
                            {"x": "y1"},
                            {"x": "y2"},
                            {"x": "y3"},
                            {"x": "y4"},
                            {"x": "y5"},
                        ],
                        "train": [
                            {"b": "y1"},
                            {"b": "y2"},
                            {"b": "y3"},
                            {"b": "y4"},
                            {"b": "y5"},
                        ],
                    }
                ),
            },
            weights={"origin1": 2, "origin2": 1},
            max_total_samples=20,
        )

        res = operator()
        targets = {
            "test": [
                {"x": "y1", "subset": ["origin2"]},
                {"x": "x1", "subset": ["origin1"]},
                {"x": "y2", "subset": ["origin2"]},
                {"x": "x2", "subset": ["origin1"]},
                {"x": "x3", "subset": ["origin1"]},
                {"x": "x4", "subset": ["origin1"]},
                {"x": "x5", "subset": ["origin1"]},
                {"x": "y3", "subset": ["origin2"]},
                {"x": "y4", "subset": ["origin2"]},
                {"x": "y5", "subset": ["origin2"]},
            ],
            "train": [
                {"a": "x1", "subset": ["origin1"]},
                {"b": "y1", "subset": ["origin2"]},
                {"a": "x2", "subset": ["origin1"]},
                {"a": "x3", "subset": ["origin1"]},
                {"a": "x4", "subset": ["origin1"]},
                {"a": "x5", "subset": ["origin1"]},
                {"b": "y2", "subset": ["origin2"]},
                {"b": "y3", "subset": ["origin2"]},
                {"b": "y4", "subset": ["origin2"]},
                {"b": "y5", "subset": ["origin2"]},
            ],
        }
        for key in ["train", "test"]:
            self.compare_stream(targets[key], list(res[key]))

        targets = [
            {"x": "x1", "subset": ["origin1"]},
            {"x": "x2", "subset": ["origin1"]},
            {"x": "x3", "subset": ["origin1"]},
            {"x": "y1", "subset": ["origin2"]},
            {"x": "y2", "subset": ["origin2"]},
            {"x": "y3", "subset": ["origin2"]},
        ]

        # check_operator(
        # operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        # )

    def test_end_to_end(self):
        dataset = WeightedFusion(
            subsets={
                "wnli": DatasetRecipe(
                    card="cards.wnli",
                    template="templates.classification.multi_class.relation.default",
                    group_by=["template"],
                ),
                "rte": DatasetRecipe(
                    card="cards.rte",
                    template="templates.classification.multi_class.relation.default",
                ),
                "stsb": WeightedFusion(
                    subsets={
                        "regression": DatasetRecipe(
                            card="cards.stsb",
                            template="templates.regression.two_texts.simple",
                        ),
                        "classification": DatasetRecipe(
                            card="cards.stsb",
                            template=[
                                "templates.regression.two_texts.similarity.flan",
                                "templates.regression.two_texts.title",
                            ],
                            group_by=["template"],
                        ),
                    },
                    weights={"regression": 1, "classification": 1},
                ),
            },
            weights={"wnli": 1, "rte": 1, "stsb": 1},
            max_total_samples=30,
        )().to_dataset()
        predictions = ["not entailment"] * 20 + ["2"] * 10
        result = evaluate(predictions=predictions, data=dataset["test"])
        self.assertEqual(
            round_values(fillna(result[0]["score"], None), 3),
            {
                "instance": {
                    "score": None,
                    "score_name": "spearmanr",
                    "spearmanr": None,
                },
                "subsets": {
                    "stsb": {
                        "classification": {
                            "num_of_instances": 5,
                            "spearmanr": -0.968,
                            "score": -0.968,
                            "spearmanr_p_value": 0.007,
                            "score_name": "spearmanr",
                            "score_ci_low": None,
                            "score_ci_high": None,
                            "spearmanr_ci_low": None,
                            "spearmanr_ci_high": None,
                            "groups": {
                                "template": {
                                    "templates.regression.two_texts.title": {
                                        "num_of_instances": 4,
                                        "spearmanr": -0.943,
                                        "spearmanr_p_value": 0.057,
                                        "score": -0.943,
                                        "score_name": "spearmanr",
                                        "score_ci_low": None,
                                        "score_ci_high": None,
                                        "spearmanr_ci_low": None,
                                        "spearmanr_ci_high": None,
                                    },
                                    "templates.regression.two_texts.similarity.flan": {
                                        "num_of_instances": 1,
                                        "spearmanr": None,
                                        "spearmanr_p_value": None,
                                        "score": None,
                                        "score_name": "spearmanr",
                                    },
                                }
                            },
                        },
                        "regression": {
                            "num_of_instances": 8,
                            "spearmanr": -0.065,
                            "score": -0.065,
                            "spearmanr_p_value": 0.879,
                            "score_name": "spearmanr",
                            "score_ci_low": None,
                            "score_ci_high": None,
                            "spearmanr_ci_low": None,
                            "spearmanr_ci_high": None,
                        },
                        "score": -0.517,
                        "score_name": "subsets_mean",
                        "num_of_instances": 13,
                    },
                    "wnli": {
                        "f1_macro": 0.357,
                        "f1_entailment": 0.0,
                        "f1_not entailment": 0.714,
                        "f1_macro_ci_low": 0.182,
                        "f1_macro_ci_high": 0.467,
                        "score_name": "f1_micro",
                        "score": 0.5,
                        "score_ci_high": 0.762,
                        "score_ci_low": 0.2,
                        "num_of_instances": 12,
                        "accuracy": 0.417,
                        "accuracy_ci_low": 0.167,
                        "accuracy_ci_high": 0.667,
                        "f1_micro": 0.5,
                        "f1_micro_ci_low": 0.2,
                        "f1_micro_ci_high": 0.762,
                        "groups": {
                            "template": {
                                "templates.classification.multi_class.relation.default": {
                                    "f1_macro": 0.357,
                                    "f1_entailment": 0.0,
                                    "f1_not entailment": 0.714,
                                    "f1_macro_ci_low": 0.182,
                                    "f1_macro_ci_high": 0.467,
                                    "score_name": "f1_micro",
                                    "score": 0.5,
                                    "score_ci_high": 0.762,
                                    "score_ci_low": 0.2,
                                    "num_of_instances": 12,
                                    "accuracy": 0.417,
                                    "accuracy_ci_low": 0.167,
                                    "accuracy_ci_high": 0.667,
                                    "f1_micro": 0.5,
                                    "f1_micro_ci_low": 0.2,
                                    "f1_micro_ci_high": 0.762,
                                }
                            }
                        },
                    },
                    "rte": {
                        "f1_macro": 0.333,
                        "f1_entailment": 0.0,
                        "f1_not entailment": 0.667,
                        "f1_macro_ci_low": 0.0,
                        "f1_macro_ci_high": 0.823,
                        "score_name": "f1_micro",
                        "score": 0.5,
                        "score_ci_high": 0.889,
                        "score_ci_low": 0.0,
                        "num_of_instances": 5,
                        "accuracy": 0.4,
                        "accuracy_ci_low": 0.0,
                        "accuracy_ci_high": 0.8,
                        "f1_micro": 0.5,
                        "f1_micro_ci_low": 0.0,
                        "f1_micro_ci_high": 0.889,
                    },
                    "score": 0.161,
                    "score_name": "subsets_mean",
                    "num_of_instances": 30,
                },
                "global": {
                    "score": 0.161,
                    "score_name": "subsets_mean",
                    "subsets_mean": 0.161,
                    "num_of_instances": 30,
                },
            },
        )
