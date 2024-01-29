import unittest

from src.unitxt.dict_utils import dict_get, dict_set


class TestDictUtils(unittest.TestCase):
    def test_simple_get(self):
        dic = {"a": 1, "b": 2}
        self.assertEqual(dict_get(dic, "a"), 1)
        self.assertEqual(dict_get(dic, "b"), 2)
        self.assertEqual(dict_get(dic, "c", not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "c")

    def test_nested_get(self):
        dic = {"a": {"b": 1, "c": 2}}
        self.assertEqual(dict_get(dic, "a/b", use_dpath=True), 1)
        self.assertEqual(dict_get(dic, "a/c", use_dpath=True), 2)
        self.assertEqual(dict_get(dic, "a/d", use_dpath=True, not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "a/d", use_dpath=True)

    def test_query_get(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        self.assertEqual(dict_get(dic, "a/*/b", use_dpath=True), [1, 2])
        dic = {"a": [{"b": 1}, {"b": 2}], "c": [{"b": 3}, {"b": 4}]}
        self.assertEqual(dict_get(dic, "*/1/b", use_dpath=True), [2, 4])

    def test_simple_set(self):
        dic = {"a": 1, "b": 2}
        dict_set(dic, "a", 3)
        self.assertDictEqual(dic, {"a": 3, "b": 2})
        dict_set(dic, "b", 4)
        self.assertDictEqual(dic, {"a": 3, "b": 4})
        with self.assertRaises(ValueError):
            dict_set(dic, "c", 5, not_exist_ok=False)
        dict_set(dic, "c", 5)
        self.assertDictEqual(dic, {"a": 3, "b": 4, "c": 5})

    def test_nested_set(self):
        dic = {"a": {"b": 1, "c": 2}}
        dict_set(dic, "a/b", 3, use_dpath=True)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 2}})
        dict_set(dic, "a/c", 4, use_dpath=True)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 4}})
        with self.assertRaises(ValueError):
            dict_set(dic, "a/d", 5, use_dpath=True, not_exist_ok=False)
        dict_set(dic, "a/d", 5, use_dpath=True)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 4, "d": 5}})

    def test_query_set(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        dict_set(dic, "a/*/b", [3, 4], use_dpath=True, set_multiple=True)
        self.assertDictEqual(dic, {"a": [{"b": 3}, {"b": 4}]})
        dic = {"a": [{"b": 1}, {"b": 2}], "c": [{"b": 3}, {"b": 4}]}
        dict_set(dic, "*/1/b", [5, 6], use_dpath=True, set_multiple=True)
        self.assertDictEqual(
            dic, {"a": [{"b": 1}, {"b": 5}], "c": [{"b": 3}, {"b": 6}]}
        )

    def test_query_set_with_multiple_non_existing(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        with self.assertRaises(ValueError):
            dict_set(dic, "a/*/c", [3, 4], use_dpath=True, set_multiple=True)
        with self.assertRaises(ValueError):
            dict_set(
                dic,
                "a/*/c",
                [3, 4],
                use_dpath=True,
                set_multiple=False,
                not_exist_ok=True,
            )

    def test_adding_one_new_field(self):
        dic = {"a": {"b": {"c": 0}}}
        dict_set(dic, "a/b/d", 1, use_dpath=True)
        self.assertDictEqual(dic, {"a": {"b": {"c": 0, "d": 1}}})

    def test_adding_one_new_field_nested(self):
        dic = {"d": 0}
        dict_set(dic, "/a/b/d", 1, use_dpath=True)
        self.assertDictEqual(dic, {"a": {"b": {"d": 1}}, "d": 0})
