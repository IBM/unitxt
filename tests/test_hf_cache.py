import tempfile
import threading
import unittest

from src.unitxt.stream import MultiStream, Stream
from src.unitxt.test_utils.environment import modified_environment
from src.unitxt.test_utils.storage import get_directory_size

threading_lock = threading.Lock()


class TestHfCache(unittest.TestCase):
    def test_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with modified_environment(HF_DATASETS_CACHE=tmp_dir):
                self.assertEqual(get_directory_size(tmp_dir), 0)

                def gen():
                    for i in range(10000000):  # must be big or else hf
                        yield {"x": i}

                ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                    disable_cache=False, cache_dir=tmp_dir
                )
                for i, item in enumerate(ds["test"]):
                    self.assertEqual(item["x"], i)
                self.assertNotEqual(get_directory_size(tmp_dir), 0)

    def test_not_caching_stream(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with modified_environment(HF_DATASETS_CACHE=tmp_dir):
                self.assertEqual(get_directory_size(tmp_dir), 0)

                def gen():
                    for i in range(3):
                        yield {"x": i}

                ds = MultiStream({"test": Stream(generator=gen)}).to_dataset(
                    cache_dir=tmp_dir
                )
                for i, item in enumerate(ds["test"]):
                    self.assertEqual(item["x"], i)
                self.assertEqual(get_directory_size(tmp_dir), 0)


if __name__ == "__main__":
    unittest.main()
