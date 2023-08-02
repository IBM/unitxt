import unittest

from src.unitxt.operators import LibraryFunctionOperator, PythonFunctionOperator
from src.unitxt.test_utils.operators import apply_operator


class TestFunctionOperators(unittest.TestCase):
    def test_python_function_operator(self):
        operator = PythonFunctionOperator(
            field="a",
            function_path="str.upper",
        )

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        targets = [
            {"a": "A"},
            {"a": "B"},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_python_function_operator_with_kwargs(self):
        operator = PythonFunctionOperator(
            field="a",
            function_path="sorted",
            kwargs={"reverse": True},
        )

        inputs = [
            {"a": [1, 2, 3]},
            {"a": [3, 1, 2]},
        ]

        targets = [
            {"a": [3, 2, 1]},
            {"a": [3, 2, 1]},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_python_function_operator_with_value_arg_name(self):
        operator = PythonFunctionOperator(
            field="a",
            function_path="sorted",
            argv_before_value=[[1, 2, 3]],
            value_arg_name="reverse",
        )

        inputs = [
            {"a": True},
            {"a": False},
        ]

        targets = [
            {"a": [3, 2, 1]},
            {"a": [1, 2, 3]},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_python_function_operator_unpack_as_list(self):
        operator = PythonFunctionOperator(
            field="a",
            function_path="sorted",
            unpack_value_as_list=True,
        )

        inputs = [
            {"a": [[1, 2, 3]]},
            {"a": [[3, 1, 2]]},
        ]

        targets = [
            {"a": [1, 2, 3]},
            {"a": [1, 2, 3]},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_python_function_operator_unpack_as_dict(self):
        operator = PythonFunctionOperator(
            field="a",
            function_path="sorted",
            unpack_value_as_dict=True,
            argv_before_value=[[1, 2, 3]],
        )

        inputs = [
            {"a": {"reverse": True}},
            {"a": {"reverse": False}},
        ]

        targets = [
            {"a": [3, 2, 1]},
            {"a": [1, 2, 3]},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)

    def test_library_function_operator(self):
        operator = LibraryFunctionOperator(
            field="a",
            library_name="os",
            function_path="path.join",
            argv_before_value=["/dir"],
        )

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        targets = [
            {"a": "/dir/a"},
            {"a": "/dir/b"},
        ]

        outputs = apply_operator(operator=operator, inputs=inputs)

        for output, target in zip(outputs, targets):
            self.assertDictEqual(output, target)
