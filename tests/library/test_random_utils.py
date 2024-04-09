import random as python_random

from unitxt.random_utils import (
    new_random_generator,
)
from unitxt.settings_utils import get_settings

from tests.utils import UnitxtTestCase

settings = get_settings()


def randomize(sub_seed: str):
    random_generator = new_random_generator(sub_seed=sub_seed)
    return tuple(
        random_generator.randint(0, 10000000000000000000000) for _ in range(100)
    )


class TestRandomUtils(UnitxtTestCase):
    def test_default_seed(self):
        a = randomize(sub_seed="42")
        b = randomize(sub_seed="43")
        c = randomize(sub_seed=str(settings.seed))
        self.assertNotEqual(a, b)
        self.assertEqual(a, c)

    def test_non_string_seed(self):
        """A test for a seed that is not a string, and is a Hashable object."""
        a = randomize(sub_seed=50)
        b = randomize(sub_seed="50")
        self.assertEqual(a, b)

    def test_get_sub_default_random_generator(self):
        sub_seed = "a"
        self.assertEqual(randomize(sub_seed), randomize(sub_seed))

    def test_separation_from_global_python_seed(self):
        rand1 = randomize(sub_seed="b")
        python_random.seed(10)
        rand2 = randomize(sub_seed="b")
        self.assertEqual(rand1, rand2)
