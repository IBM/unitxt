from src.unitxt.string_operators import Join, Split
from src.unitxt.test_utils.operators import check_operator
from tests.utils import UnitxtTestCase


class TestStringOperators(UnitxtTestCase):
    def test_split(self):
        operator = Split(field="text", by=",")
        inputs = [{"text": "kk,ll"}]
        targets = [{"text": ["kk", "ll"]}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_join(self):
        operator = Join(field="text", by=",")
        inputs = [{"text": ["kk", "ll"]}]
        targets = [{"text": "kk,ll"}]
        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
