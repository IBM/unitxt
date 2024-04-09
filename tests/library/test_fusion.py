from unitxt.fusion import FixedFusion, WeightedFusion
from unitxt.operators import IterableSource
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestFusion(UnitxtTestCase):
    def test_unbounded_fixed_fusion(self):
        operator = FixedFusion(
            origins=[
                IterableSource({"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
        )

        targets = [
            {"x": "x1"},
            {"x": "x2"},
            {"x": "x3"},
            {"x": "y1"},
            {"x": "y2"},
            {"x": "y3"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

    def compare_stream(self, stream, expected_stream):
        self.assertEqual(len(stream), len(expected_stream))
        for input_dict, output_dict in zip(stream, expected_stream):
            self.assertDictEqual(input_dict, output_dict)

    def test_nonoverlapping_splits_fusion(self):
        operator = FixedFusion(
            origins=[
                IterableSource({"train": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
        )

        output_multi_stream = operator()
        self.assertListEqual(sorted(output_multi_stream.keys()), ["test", "train"])
        self.compare_stream(
            list(output_multi_stream["train"]), [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]
        )
        self.compare_stream(
            list(output_multi_stream["test"]), [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]
        )

    def test_bounded_fixed_fusion(self):
        operator = FixedFusion(
            origins=[
                IterableSource({"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
            max_instances_per_origin=2,
        )

        targets = [
            {"x": "x1"},
            {"x": "x2"},
            {"x": "y1"},
            {"x": "y2"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

    def test_over_bounded_fixed_fusion(self):
        operator = FixedFusion(
            origins=[
                IterableSource({"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
            max_instances_per_origin=10,
        )

        targets = [
            {"x": "x1"},
            {"x": "x2"},
            {"x": "x3"},
            {"x": "y1"},
            {"x": "y2"},
            {"x": "y3"},
        ]

        check_operator(operator, inputs=None, targets=targets, tester=self)

    def test_unbounded_weighted_fusion(self):
        operator = WeightedFusion(
            origins=[
                IterableSource({"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
            weights=[1, 1],
        )

        targets = [
            {"x": "x1"},
            {"x": "x2"},
            {"x": "x3"},
            {"x": "y1"},
            {"x": "y2"},
            {"x": "y3"},
        ]

        check_operator(
            operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        )

    def test_over_bounded_weighted_fusion(self):
        operator = WeightedFusion(
            origins=[
                IterableSource({"test": [{"x": "x1"}, {"x": "x2"}, {"x": "x3"}]}),
                IterableSource({"test": [{"x": "y1"}, {"x": "y2"}, {"x": "y3"}]}),
            ],
            weights=[1, 1],
            max_total_examples=10,
        )

        targets = [
            {"x": "x1"},
            {"x": "x2"},
            {"x": "x3"},
            {"x": "y1"},
            {"x": "y2"},
            {"x": "y3"},
        ]

        check_operator(
            operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x"
        )
