from unitxt.fusion import FixedFusion, WeightedFusion
from unitxt.operators import IterableSource
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

        # test save past groups
        operator = FixedFusion(
            origins={
                "targets1": IterableSource({"test": targets}),
                "origin3": IterableSource(
                    {"test": [{"x": "z1"}, {"x": "z2"}, {"x": "z3"}]}
                ),
            },
        )

        targets2 = [
            {"x": "x1", "group": "targets1", "past_groups": "origin1"},
            {"x": "x2", "group": "targets1", "past_groups": "origin1"},
            {"x": "x3", "group": "targets1", "past_groups": "origin1"},
            {"x": "y1", "group": "targets1", "past_groups": "origin2"},
            {"x": "y2", "group": "targets1", "past_groups": "origin2"},
            {"x": "y3", "group": "targets1", "past_groups": "origin2"},
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
        operator = FixedFusion(
            origins={
                "origin1": IterableSource(
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            max_instances_per_origin=2,
        )

        targets = [
            {"x": "x1", "group": "origin1"},
            {"x": "x2", "group": "origin1"},
            {"x": "y1", "group": "origin2"},
            {"x": "y2", "group": "origin2"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

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
            max_instances_per_origin=10,
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
                    {"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}
                ),
                "origin2": IterableSource(
                    {"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}
                ),
            },
            weights={"origin1": 1, "origin2": 1},
            max_total_examples=10,
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
