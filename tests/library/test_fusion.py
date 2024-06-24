from unitxt.api import evaluate
from unitxt.fusion import FixedFusion, WeightedFusion
from unitxt.operators import IterableSource
from unitxt.standard import StandardRecipe
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestFusion(UnitxtTestCase):
    def test_unbounded_fixed_fusion(self):
        operator = FixedFusion(
            origins={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
        )

        targets = [
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "x3", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
            {"x": "y3", "group": "origin2"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

        # test the saving of past groups
        origin3 = [{"x": "z1"}, {"x": "z2"}, {"x": "z3"}]
        operator = FixedFusion(
            origins={
                "targets1": IterableSource({"test": targets}),
                "origin3": IterableSource({"test": origin3}),
            },
        )

        targets2 = [
            {"x": "x1", "group": "targets1/origin1"},
            {"x": "x2", "group": "targets1/origin1"},
            {"x": "x3", "group": "targets1/origin1"},
            {"x": "y1", "group": "targets1/origin2"},
            {"x": "y2", "group": "targets1/origin2"},
            {"x": "y3", "group": "targets1/origin2"},
            {"x": "z1", "group": "origin3"},
            {"x": "z2", "group": "origin3"},
            {"x": "z3", "group": "origin3"},
        ]
        check_operator(operator, inputs=None, targets=targets2, tester=self)

    def compare_stream(self, stream, expected_stream):
        self.assertEqual(len(stream), len(expected_stream))
        for input_dict, output_dict in zip(stream, expected_stream):
            self.assertDictEqual(input_dict, output_dict)

    def test_nonoverlapping_splits_fusion(self):
        operator = FixedFusion(
            origins={
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
                {"x": "x1", "group": "origin_train"},
                {"x": "x2", "group": "origin_train"},
                {"x": "x3", "group": "origin_train"},
            ],
        )
        self.compare_stream(
            list(output_multi_stream["test"]),
            [
                {"x": "y1", "group": "origin_test"},
                {"x": "y2", "group": "origin_test"},
                {"x": "y3", "group": "origin_test"},
            ],
        )

    def test_bounded_fixed_fusion(self):
        fixed_fusion = FixedFusion(
            origins={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            max_instances_per_origin_split=2,
        )
        targets = [
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
        ]
        outputs = fixed_fusion()
        self.compare_stream(targets, list(outputs["test"]))

    def test_over_bounded_fixed_fusion(self):
        operator = FixedFusion(
            origins={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            max_instances_per_origin_split=10,
        )

        targets = [
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "x3", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
            {"x": "y3", "group": "origin2"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

    def test_unbounded_weighted_fusion(self):
        operator = WeightedFusion(
            origins={
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
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "x3", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
            {"x": "y3", "group": "origin2"},
        ]

        check_operator(
            operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        )

    def test_over_bounded_weighted_fusion(self):
        operator = WeightedFusion(
            origins={
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
            max_total_examples=3,
        )

        res = operator()
        targets = {
            "test": [
                {"x": "y1", "group": "origin2"},
                {"x": "x1", "group": "origin1"},
                {"x": "y2", "group": "origin2"},
            ],
            "train": [
                {"a": "x1", "group": "origin1"},
                {"b": "y1", "group": "origin2"},
                {"b": "y2", "group": "origin2"},
            ],
        }
        for key in res:
            self.compare_stream(targets[key], list(res[key]))

        operator = WeightedFusion(
            origins={
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
            max_total_examples=20,
        )

        res = operator()
        targets = {
            "test": [
                {"x": "y1", "group": "origin2"},
                {"x": "x1", "group": "origin1"},
                {"x": "y2", "group": "origin2"},
                {"x": "x2", "group": "origin1"},
                {"x": "x3", "group": "origin1"},
                {"x": "x4", "group": "origin1"},
                {"x": "x5", "group": "origin1"},
                {"x": "y3", "group": "origin2"},
                {"x": "y4", "group": "origin2"},
                {"x": "y5", "group": "origin2"},
            ],
            "train": [
                {"a": "x1", "group": "origin1"},
                {"b": "y1", "group": "origin2"},
                {"a": "x2", "group": "origin1"},
                {"a": "x3", "group": "origin1"},
                {"a": "x4", "group": "origin1"},
                {"a": "x5", "group": "origin1"},
                {"b": "y2", "group": "origin2"},
                {"b": "y3", "group": "origin2"},
                {"b": "y4", "group": "origin2"},
                {"b": "y5", "group": "origin2"},
            ],
        }
        for key in res:
            self.compare_stream(targets[key], list(res[key]))

        targets = [
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "x3", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
            {"x": "y3", "group": "origin2"},
        ]

        # check_operator(
        # operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        # )

    def test_end_to_end(self):
        dataset = WeightedFusion(
            origins={
                "wnli": StandardRecipe(
                    card="cards.wnli",
                    template="templates.classification.multi_class.relation.default",
                ),
                "rte": StandardRecipe(
                    card="cards.rte",
                    template="templates.classification.multi_class.relation.default",
                ),
                "stsb": StandardRecipe(
                    card="cards.stsb", template="templates.regression.two_texts.title"
                ),
            },
            weights={"wnli": 1, "rte": 1, "stsb": 1},
            max_total_examples=30,
        )().to_dataset()
        predictions = ["not entailment"] * 20 + ["2"] * 10
        result = evaluate(predictions=predictions, data=dataset["test"])

        self.assertEqual(result[0]["score"]["global"]["rte"]["score"], 0.5)
        self.assertEqual(result[0]["score"]["global"]["wnli"]["score"], 0.5)
        self.assertAlmostEqual(
            result[0]["score"]["global"]["stsb"]["score"], 0.046, places=3
        )
        self.assertAlmostEqual(result[0]["score"]["global"]["score"], 0.349, places=3)
