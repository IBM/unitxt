from src.unitxt.collections_operators import (
    Dictify,
    DuplicateByList,
    DuplicateBySubLists,
    Get,
    Slice,
    Wrap,
)
from src.unitxt.test_utils.operators import check_operator
from tests.utils import UnitxtTestCase


class TestCollectionsOperators(UnitxtTestCase):
    def test_dictify(self):
        operator = Dictify(field="tuple", with_keys=["a", "b"], to_field="dict")

        inputs = [{"tuple": (1, 2)}]

        targets = [{"tuple": (1, 2), "dict": {"a": 1, "b": 2}}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_wrap(self):
        operator = Wrap(field="x", inside="tuple")

        inputs = [{"x": 1}]

        targets = [{"x": (1,)}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

        with self.assertRaises(ValueError):
            Wrap(field="x", inside="non_existing_collection")

    def test_slice(self):
        operator = Slice(field="x", start=1, stop=3)

        inputs = [{"x": [0, 1, 2, 3, 4]}]

        targets = [{"x": [1, 2]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_get(self):
        operator = Get(field="x", item=-1)

        inputs = [{"x": [0, 1, 2, 3, 4]}]

        targets = [{"x": 4}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list(self):
        operator = DuplicateByList(field="x", to_field="y")

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [
            {"y": 0, "x": [0, 1, 2]},
            {"y": 1, "x": [0, 1, 2]},
            {"y": 2, "x": [0, 1, 2]},
            {"y": 3, "x": [3]},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list_same_field(self):
        operator = DuplicateByList(field="x")

        inputs = [{"x": [0, 1]}]

        targets = [
            {"x": 0},
            {"x": 1},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_list_deep_copy(self):
        operator = DuplicateByList(field="x", use_deep_copy=True)

        inputs = [{"x": [0, 1]}]

        targets = [
            {"x": 0},
            {"x": 1},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_sub_lists(self):
        operator = DuplicateBySubLists(field="x")

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [{"x": [0]}, {"x": [0, 1]}, {"x": [0, 1, 2]}, {"x": [3]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_duplicate_by_sub_lists_with_deep_copy(self):
        operator = DuplicateBySubLists(field="x", use_deep_copy=True)

        inputs = [{"x": [0, 1, 2]}, {"x": []}, {"x": [3]}]

        targets = [{"x": [0]}, {"x": [0, 1]}, {"x": [0, 1, 2]}, {"x": [3]}]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
