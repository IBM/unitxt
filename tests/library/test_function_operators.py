import json
import os
import tempfile
import types

from unitxt.artifact import Artifact
from unitxt.operator import SequentialOperator
from unitxt.operators import Apply, CopyFields, FunctionOperator
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


def process_stream(stream, stream_name=None):
    for instance in stream:
        instance["x"] += 1
        yield instance


def process_instance(instance, stream_name=None):
    instance["x"] += 1
    return instance


def wrong_function(instance):
    ...


class TestFunctionOperators(UnitxtTestCase):
    def test_saving_and_loading_operator_holding_function_operator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "temp_func.json")
            SequentialOperator(steps=[process_stream]).save(artifact_path)

            loaded = Artifact.load(artifact_path)
        self.assertIsInstance(loaded, SequentialOperator)
        if isinstance(loaded, SequentialOperator):
            self.assertIsInstance(loaded.steps[0], FunctionOperator)
            if isinstance(loaded.steps[0], FunctionOperator):
                self.assertIsInstance(loaded.steps[0].function, types.FunctionType)

    def test_saving_and_loading_function_operator(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "temp_func.json")
            FunctionOperator(function=process_stream).save(artifact_path)

            loaded = Artifact.load(artifact_path)
        self.assertIsInstance(loaded, FunctionOperator)
        if isinstance(loaded, FunctionOperator):
            self.assertIsInstance(loaded.function, types.FunctionType)

    def test_saving_and_loading_operator_with_regular_function(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_path = os.path.join(temp_dir, "temp_func.json")
            SequentialOperator(steps=[wrong_function]).save(artifact_path)

            loaded = Artifact.load(artifact_path)
        self.assertIsInstance(loaded, SequentialOperator)
        if isinstance(loaded, SequentialOperator):
            self.assertIsInstance(loaded.steps[0], types.FunctionType)

    def test_stream_function_operators(self):
        operator = FunctionOperator(function=process_stream)

        inputs = [
            {"x": 1, "b": "2"},
            {"x": 2, "b": "3"},
        ]

        targets = [
            {"x": 2, "b": "2"},
            {"x": 3, "b": "3"},
        ]

        check_operator(
            operator=operator,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_instance_function_operators(self):
        operator = FunctionOperator(function=process_instance)

        inputs = [
            {"x": 1, "b": "2"},
            {"x": 2, "b": "3"},
        ]

        targets = [
            {"x": 2, "b": "2"},
            {"x": 3, "b": "3"},
        ]

        check_operator(
            operator=operator,
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_function_operator_with_wrong_function(self):
        with self.assertRaises(ValueError):
            FunctionOperator(function=[])
        with self.assertRaises(TypeError):
            FunctionOperator(function=wrong_function)

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
