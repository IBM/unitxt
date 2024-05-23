import copy
import time

from unitxt.operators import Apply
from unitxt.stream import Stream
from unitxt.test_utils.operators import apply_operator

from tests.utils import UnitxtTestCase


class TestStreamCaching(UnitxtTestCase):
    def test_non_caching_stream(self):
        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=False,
        )

        self.assertNotEqual(next(iter(stream))["x"], next(iter(stream))["x"])

    def test_caching_stream(self):
        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=True,
        )

        self.assertEqual(next(iter(stream))["x"], next(iter(stream))["x"])

    def test_operator_not_caching(self):
        operator = Apply(function=time.time, to_field="b", caching=False)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        targets1 = apply_operator(operator=operator, inputs=copy.deepcopy(inputs))
        targets2 = apply_operator(operator=operator, inputs=copy.deepcopy(inputs))

        for target1, target2 in zip(targets1, targets2):
            self.assertNotEqual(target1["b"], target2["b"])

    def test_operator_caching(self):
        operator = Apply(function=time.time, to_field="b", caching=True)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        apply_operator(operator=operator, inputs=copy.deepcopy(inputs))

        # check_operator(operator=operator, inputs=copy.deepcopy(inputs), targets=targets, tester=self)
