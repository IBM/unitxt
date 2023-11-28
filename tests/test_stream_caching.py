import copy
import time
import unittest

from src.unitxt.operators import Apply
from src.unitxt.stream import MultiStream, Stream
from src.unitxt.test_utils.operators import apply_operator, check_operator


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

    def test_operator_caching(self):
        operator = Apply(function=time.time, to_field="a", caching=True)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        multi_stream_input = MultiStream.from_iterables({"test": inputs}, copying=True)
        multi_stream = operator(multi_stream_input)
        output1 = list(multi_stream["test"])
        output2 = list(multi_stream["test"])

        self.assertEqual(output1, output2)

    def test_operator_caching_on_disk(self):
        operator = Apply(function=time.time, to_field="a", caching=False)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        multi_stream_input = MultiStream.from_iterables({"test": inputs}, copying=True)
        multi_stream = operator(multi_stream_input)
        for stream in multi_stream.values():
            stream.caching = True
            stream.cache_on_disk = True
        output1 = list(multi_stream["test"])
        output2 = list(multi_stream["test"])

        self.assertEqual(output1, output2)

    def test_operator_not_caching(self):

        operator = Apply(function=time.time, to_field="a", caching=False)

        inputs = [
            {"a": "a"},
            {"a": "b"},
        ]

        multi_stream_input = MultiStream.from_iterables({"test": inputs}, copying=True)
        multi_stream = operator(multi_stream_input)
        output1 = list(multi_stream["test"])
        output2 = list(multi_stream["test"])

        self.assertNotEqual(output1, output2)
