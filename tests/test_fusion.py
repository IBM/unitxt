import json
import unittest

from src.unitxt.fusion import FixedFusion, WeightedFusion
from src.unitxt.operators import IterableSource
from src.unitxt.test_utils.operators import test_operator


class TestFusion(unittest.TestCase):
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

        test_operator(operator, inputs=None, targets=targets, tester=self)

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

        test_operator(operator, inputs=None, targets=targets, tester=self)

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

        test_operator(operator, inputs=None, targets=targets, tester=self)

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

        test_operator(operator, inputs=None, targets=targets, tester=self, sort_outputs_by="x")
