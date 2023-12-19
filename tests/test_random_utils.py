import random as python_random
import unittest

from src.unitxt.random_utils import (
    __default_seed__,
    get_sub_default_random_generator,
)


def randomize(random_generator):
    return tuple(
        random_generator.randint(0, 10000000000000000000000) for _ in range(100)
    )


class TestRandomUtils(unittest.TestCase):
    def test_default_seed(self):
        a = randomize(get_sub_default_random_generator(sub_seed="42"))
        b = randomize(get_sub_default_random_generator(sub_seed="43"))
        c = randomize(get_sub_default_random_generator(sub_seed=str(__default_seed__)))
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def compare_get_sub_default_random_generator_with_same_seed(self, sub_seed: str):
        rand1 = get_sub_default_random_generator(sub_seed=sub_seed)
        rand2 = get_sub_default_random_generator(sub_seed=sub_seed)
        self.assertEqual(randomize(rand1), randomize(rand2))

    def test_get_sub_default_random_generator(self):
        self.compare_get_sub_default_random_generator_with_same_seed(sub_seed="a")

    def test_separation_from_global_python_seed(self):
        rand1 = get_sub_default_random_generator(sub_seed="b")
        python_random.seed(10)
        rand2 = get_sub_default_random_generator(sub_seed="b")
        self.assertEqual(randomize(rand1), randomize(rand2))

    @staticmethod
    def thread_function(name, sleep_time, results):
        import time

        time.sleep(sleep_time)
        rand1 = get_sub_default_random_generator(sub_seed="b")
        results[name][0] = randomize(rand1)

        time.sleep(sleep_time)

        rand2 = get_sub_default_random_generator(sub_seed="b")
        results[name][1] = randomize(rand2)

    def test_thread_safety_sanity(self):
        results = []
        for i in range(3):
            sleep_time = python_random.randint(0, 100) / 10000
            results.append([None, None])
            TestRandomUtils.thread_function(i, sleep_time, results)
            # x = threading.Thread(target=thread_function, args=(i, sleep_time, results))
            # threads.append(x)
            # x.start()

        for index in range(3):
            with self.subTest(f"Within Thread {index}"):
                self.assertEqual(results[index][0], results[index][1])

        with self.subTest("Across all threads"):
            flatten_results = [item for sublist in results for item in sublist]
            self.assertEqual(len(set(flatten_results)), 1)

    def test_thread_safety(self):
        import threading

        threads = []
        results = []
        for i in range(1):
            sleep_time = python_random.randint(0, 100) / 10000
            results.append([None, None])
            x = threading.Thread(
                target=TestRandomUtils.thread_function, args=(i, sleep_time, results)
            )
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()

            with self.subTest(f"Within Thread {index}"):
                self.assertIsNotNone(results[index][0])
                self.assertIsNotNone(results[index][1])
                self.assertEqual(results[index][0], results[index][1])

        with self.subTest("Across all threads"):
            flatten_results = [item for sublist in results for item in sublist]
            self.assertEqual(len(set(flatten_results)), 1)
