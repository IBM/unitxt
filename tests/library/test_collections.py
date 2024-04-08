from unitxt.collections import DictCollection, ListCollection

from tests.utils import UnitxtTestCase


class TestCollections(UnitxtTestCase):
    def test_dict_collection(self):
        c = DictCollection({0: 1})
        self.assertEqual(c[0], 1)
        with self.assertRaises(LookupError):
            c[1]

    def test_list_collection(self):
        c = ListCollection([1])
        self.assertEqual(c[0], 1)
        with self.assertRaises(LookupError):
            c[1]
