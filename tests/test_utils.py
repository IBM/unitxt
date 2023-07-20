import unittest
from src.unitxt.utils import dict_query

d = {'a': {'b': 1}, 'c': [{'d': 2}, {'d': 3}], 'e': [1, 2, 3], 'f': 0}


class DictQueryTest(unittest.TestCase):

    def test_simple_access(self):
        self.assertEqual(dict_query(d, 'f'), d['f'], 0)

    def test_second_level_access(self):
        self.assertEqual(dict_query(d, 'a/b'), d['a']['b'], 1)

    def test_first_level_access(self):
        self.assertEqual(dict_query(d, 'a'), d['a'], {'b': 1})

    def test_wildcard_access(self):
        self.assertEqual(dict_query(d, 'c/*/d'), [t['d'] for t in d['c']], [2, 3])

    def test_int_access(self):
        self.assertEqual(dict_query(d, 'c/0/d'), d['c'][0]['d'], 2)

    def test_first_list_access(self):
        self.assertEqual(dict_query(d, 'e/0'), d['e'][0], 1)

    def test_second_list_access(self):
        self.assertEqual(dict_query(d, 'e/*'), d['e'], [1, 2, 3])

    def test_list_access(self):
        self.assertEqual(dict_query(d, 'e'), d['e'], [1, 2, 3])

