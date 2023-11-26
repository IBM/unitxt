import random as python_random
import unittest

from src.unitxt.random_utils import __default_seed__, nested_seed, random, set_seed


def first_randomization():
    return tuple(random.randint(0, 10000000000000000000000) for _ in range(100))


class TestRandomUtils(unittest.TestCase):
    def test_default_seed(self):
        set_seed(42)
        a = first_randomization()
        set_seed(43)
        b = first_randomization()
        set_seed(__default_seed__)
        c = first_randomization()
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def test_nested_level_difference(self):
        with nested_seed():
            with nested_seed():
                a = first_randomization()

        with nested_seed():
            b = first_randomization()

        self.assertNotEqual(a, b)

    def test_nested_level_similarity(self):
        with nested_seed():
            with nested_seed():
                a = first_randomization()

        with nested_seed():
            with nested_seed():
                b = first_randomization()

        self.assertEqual(a, b)

    def test_sepration_from_global_python_seed(self):
        with nested_seed():
            with nested_seed():
                a = first_randomization()

        with nested_seed():
            with nested_seed():
                python_random.seed(10)
                b = first_randomization()

        self.assertEqual(a, b)

    def test_thread_safety(self):
        import threading
        import time

        def thread_function(name, sleep_time, results):
            time.sleep(sleep_time)

            time.sleep(sleep_time)
            with nested_seed():
                time.sleep(sleep_time)
                with nested_seed():
                    time.sleep(sleep_time)
                    a = first_randomization()
            results[name][0] = a

            time.sleep(sleep_time)
            with nested_seed():
                time.sleep(sleep_time)
                with nested_seed():
                    time.sleep(sleep_time)
                    b = first_randomization()
            results[name][1] = b

        threads = list()
        results = list()
        for i in range(100):
            sleep_time = python_random.randint(0, 100) / 1000
            results.append([None, None])
            x = threading.Thread(target=thread_function, args=(i, sleep_time, results))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()
            self.assertEqual(results[index][0], results[index][1])

        flatten_results = [item for sublist in results for item in sublist]
        self.assertEqual(len(set(flatten_results)), 1)
