import time
import unittest

from src.unitxt.operators import Apply
from src.unitxt.stream import Stream
from src.unitxt.test_utils.operators import apply_operator, test_operator


class TestStreamCaching(unittest.TestCase):
    def test_non_caching_stream(self):
        original_list = list(range(1000))
        original_list.append(0)

        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=False,
        )

        self.assertNotEqual(list(stream)[0]["x"], list(stream)[0]["x"])

    def test_caching_stream(self):
        original_list = list(range(1000))
        original_list.append(0)

        def generator():
            yield {"x": str(time.time())}

        stream = Stream(
            generator=generator,
            caching=True,
        )

        self.assertEqual(list(stream)[0]["x"], list(stream)[0]["x"])

    def test_operator_caching(self):
        operator = Apply(function=time.time, to_field="b", caching=True)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        targets = apply_operator(operator=operator, inputs=inputs)

        test_operator(operator=operator, inputs=inputs, targets=targets, tester=self)
