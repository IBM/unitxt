import unittest

from src.unitxt.operators import Apply
from src.unitxt.test_utils.operators import apply_operator


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

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_apply_function_operator_serialization(self):
        operator = Apply("a", function=str.upper, to_field="b")
        dic = operator.get_init_dict()

        self.assertDictEqual(
            {"type": "apply", "function": "str.upper", "to_field": "b", "_argv": ("a",), "_kwargs": {}}, dic
        )
