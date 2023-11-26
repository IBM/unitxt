import copy
import time
import unittest

from src.unitxt.operators import Apply
from src.unitxt.stream import Stream
from src.unitxt.test_utils.operators import apply_operator


class TestStreamCaching(unittest.TestCase):
    def test_non_caching_stream(self):
        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=False,
        )

        self.assertNotEqual(list(stream)[0]["x"], list(stream)[0]["x"])

    def test_caching_stream(self):
        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=True,
        )

        self.assertEqual(list(stream)[0]["x"], list(stream)[0]["x"])

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

        _ = apply_operator(operator=operator, inputs=copy.deepcopy(inputs))

        # test_operator(operator=operator, inputs=copy.deepcopy(inputs), targets=targets, tester=self)
