import tempfile
import unittest

from unitxt.stream import MultiStream, Stream
from unitxt.test_utils.storage import get_directory_size

from tests.utils import UnitxtTestCase


class TestHfCache(UnitxtTestCase):
    def test_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(get_directory_size(tmp_dir), 0)

            def gen():
                for i in range(100000):  # must be big or else hf won't cache
                    yield {"x": i}

            ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                disable_cache=False, cache_dir=tmp_dir
            )
            for i, item in enumerate(ds["test"]):
                self.assertEqual(item["x"], i)
            self.assertNotEqual(get_directory_size(tmp_dir), 0)

    def test_not_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(get_directory_size(tmp_dir), 0)

            def gen():
                for i in range(100000):  # must be big or else hf won't cache
                    yield {"x": i}

            ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                cache_dir=tmp_dir
            )
            for i, item in enumerate(ds["test"]):
                self.assertEqual(item["x"], i)
            self.assertEqual(get_directory_size(tmp_dir), 0)


if __name__ == "__main__":
    unittest.main()
