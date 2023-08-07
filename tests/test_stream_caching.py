import time
import unittest

from src.unitxt.stream import Stream


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
