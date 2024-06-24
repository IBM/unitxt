import json

from unitxt.operator import SequentialOperator
from unitxt.operators import Apply, CopyFields
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class TestFunctionOperators(UnitxtTestCase):
    def test_apply_function_operator(self):
        operator = Apply("a", function=str.upper, to_field="b")

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        targets = [
            {"a": "a", "b": "A"},
            {"a": "b", "b": "B"},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_apply_function_operator_for_library_function(self):
        operator = SequentialOperator(
            steps=[
                Apply(function=dict, to_field="t"),
                CopyFields(field_to_field={"a": "t/a", "b": "t/b"}),
                Apply("t", function=json.dumps, to_field="t"),
            ]
        )

        inputs = [
            {"a": "a", "b": "A"},
            {"a": "b", "b": "B"},
        ]

        targets = [
            {"a": "a", "b": "A", "t": '{"a": "a", "b": "A"}'},
            {"a": "b", "b": "B", "t": '{"a": "b", "b": "B"}'},
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_apply_function_operator_serialization(self):
        operator = Apply("a", function=str.upper, to_field="b")
        dic = operator.to_dict()

        self.assertDictEqual(
            {
                "__type__": "apply",
                "function": "str.upper",
                "to_field": "b",
                "_argv": ("a",),
            },
            dic,
        )
