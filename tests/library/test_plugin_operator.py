from src.unitxt.operator import PlugInOperator
from src.unitxt.test_utils.operators import (
    check_operator,
)
from tests.utils import UnitxtTestCase

inputs = [
    {
        "prediction": "abc",
        "references": [],
        "b": 2,
        "c": "processors.first_character",
    },
    {
        "prediction": "cba",
        "references": [],
        "b": 3,
        "c": "processors.first_character",
    },
]

targets = [
    {
        "prediction": "a",
        "references": [],
        "b": 2,
        "c": "processors.first_character",
    },
    {
        "prediction": "c",
        "references": [],
        "b": 3,
        "c": "processors.first_character",
    },
]


class TestOperators(UnitxtTestCase):
    def test_plugin_operator(self):
        check_operator(
            operator=PlugInOperator(field="c"),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_plugin_operator_with_default_opearator(self):
        check_operator(
            operator=PlugInOperator(
                field="not_a_field", default="processors.first_character"
            ),
            inputs=inputs,
            targets=targets,
            tester=self,
        )

    def test_plugin_operator_without_plug(self):
        check_operator(
            operator=PlugInOperator(field="not_a_field"),
            inputs=inputs,
            targets=inputs,
            tester=self,
        )
