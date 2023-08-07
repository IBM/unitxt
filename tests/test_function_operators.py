import json
import unittest

from src.unitxt.operator import SequntialOperator
from src.unitxt.operators import AddFields, Apply, CopyFields
from src.unitxt.test_utils.operators import test_operator


class TestFunctionOperators(unittest.TestCase):
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

        test_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_apply_function_operator_for_library_function(self):
        operator = SequntialOperator(
            [
                Apply(function=dict, to_field="t"),
                CopyFields(field_to_field={"a": "t/a", "b": "t/b"}, use_query=True),
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

        test_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_apply_function_operator_serialization(self):
        operator = Apply("a", function=str.upper, to_field="b")
        dic = operator.to_dict()

        self.assertDictEqual({"type": "apply", "function": "str.upper", "to_field": "b", "_argv": ("a",)}, dic)
