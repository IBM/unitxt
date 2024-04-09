from unitxt.dict_utils import dict_delete, dict_get, dict_set

from tests.utils import UnitxtTestCase


class TestDictUtils(UnitxtTestCase):
    def test_simple_get(self):
        dic = {
            "a": 1,
            "b": 2,
            "d": [3, 4],
            "f": [],
            "g": [[]],
            "h": [[[]]],
            "i": [[], []],
            "j": {},
            "k": [{}, {}],
            "l": [{"l1": 100}, {"l2": 200}],
        }
        self.assertEqual(dict_get(dic, "a"), 1)
        self.assertEqual(dict_get(dic, "b"), 2)
        self.assertEqual(dict_get(dic, "c", not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "c")
        self.assertEqual(dict_get(dic, "d"), [3, 4])
        self.assertEqual(dict_get(dic, "d/1"), 4)
        self.assertEqual(dict_get(dic, "d/8", not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "d/2")
        self.assertEqual(dict_get(dic, "f"), [])
        self.assertEqual(dict_get(dic, "f/*"), [])
        self.assertEqual(dict_get(dic, "f/0", not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "f/0")
        with self.assertRaises(ValueError):
            dict_get(dic, "f/*/*")
        self.assertEqual(dict_get(dic, "g/"), [[]])
        self.assertEqual(dict_get(dic, "g/*"), [[]])
        self.assertEqual(dict_get(dic, "g/*/*"), [[]])
        self.assertEqual(dict_get(dic, "g/0"), [])
        with self.assertRaises(ValueError):
            dict_get(dic, "g/*/*/*")
        self.assertEqual(dict_get(dic, "g/*/*"), dict_get(dic, "h/0"))
        self.assertEqual(dict_get(dic, "i/0"), dict_get(dic, "i/1"))
        self.assertEqual({}, dict_get(dic, "j"))
        self.assertEqual([], dict_get(dic, "j/*"))
        self.assertEqual([{}, {}], dict_get(dic, "k"))
        self.assertEqual([{}, {}], dict_get(dic, "k/*"))
        self.assertEqual([[], []], dict_get(dic, "k/*/*"))
        with self.assertRaises(ValueError):
            dict_get(dic, "k/*/*/*")
        self.assertEqual([{"l1": 100}, {"l2": 200}], dict_get(dic, "l/*"))
        self.assertEqual([[100], [200]], dict_get(dic, "l/*/*"))
        with self.assertRaises(ValueError):
            dict_get(dic, "l/*/*/*")

    def test_nested_get(self):
        dic = {"x/y": 3}
        self.assertEqual(dict_get(dic, "x/y"), 3)

        dic = {"a": {"b": 1, "c": 2, "f": [3, 4], "g": []}}
        self.assertEqual(dict_get(dic, "a/b"), 1)
        self.assertEqual(dict_get(dic, "a/c"), 2)
        self.assertEqual(dict_get(dic, "a/d", not_exist_ok=True), None)
        with self.assertRaises(ValueError):
            dict_get(dic, "a/d")
        self.assertEqual(dict_get(dic, "a/f"), [3, 4])
        self.assertEqual(dict_get(dic, "a/f/0"), 3)
        self.assertEqual(dict_get(dic, "a/f/1"), 4)
        self.assertEqual(dict_get(dic, "a/f/*"), [3, 4])
        with self.assertRaises(ValueError):
            dict_get(dic, "a/f/2")
        with self.assertRaises(ValueError):
            dict_get(dic, "a/f/1/m")
        self.assertEqual(dict_get(dic, "a/g"), [])
        self.assertEqual(dict_get(dic, "a/g/*"), [])
        with self.assertRaises(ValueError):
            dict_get(dic, "a/g/0")

    def test_query_get(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        self.assertEqual(dict_get(dic, "a/*/b"), [1, 2])
        dic = {"a": [{"b": 1}, {"b": 2}], "c": [{"b": 3}, {"b": 4}]}
        self.assertEqual(dict_get(dic, "*/1/b"), [2, 4])
        dic = {"references": ["r1", "r2", "r3"]}
        self.assertEqual(dict_get(dic, "references/*"), ["r1", "r2", "r3"])
        self.assertEqual(dict_get(dic, "references/"), ["r1", "r2", "r3"])
        self.assertEqual(dict_get(dic, "references"), ["r1", "r2", "r3"])
        with self.assertRaises(ValueError):
            dict_get(dic, "references+#/^!")

    def test_query_delete(self):
        dic = None
        with self.assertRaises(ValueError):
            dict_delete(dic, "path")
        dic = 7
        with self.assertRaises(ValueError):
            dict_delete(dic, "path")
        dic = {"a": 1, "b": 2, "c": 3}
        dict_delete(dic, "a")
        self.assertDictEqual({"b": 2, "c": 3}, dic)
        dic = {"x/y": 1, "b": 2, "c": 3}
        dict_delete(dic, "x/y")
        self.assertDictEqual({"b": 2, "c": 3}, dic)

        dic = {"a": [{"b": 1}, {"b": 2, "c": 3}]}
        dict_delete(dic, "a/*/b")
        self.assertEqual({"a": [{}, {"c": 3}]}, dic)
        dic = {"a": [{"b": 1}, {"b": 2, "c": 3}], "d": 4}
        dict_delete(dic, "a/*/b", remove_empty_ancestors=True)
        self.assertEqual({"a": [{"c": 3}], "d": 4}, dic)
        dict_delete(dic, "a/*/c", remove_empty_ancestors=True)
        self.assertEqual({"d": 4}, dic)
        dict_delete(dic, "d", remove_empty_ancestors=True)
        self.assertEqual({}, dic)
        dic = {"c": {"d": {"e": 4}}}
        dict_delete(dic, "c/d/e", remove_empty_ancestors=True)
        self.assertEqual({}, dic)
        dic = {
            "a": [
                {"b": 1},
                {"b": 2, "c": 3},
                {"b": 20, "c": 30},
                {"d": 2, "c": 5},
                {"b": 200, "c": 300},
            ]
        }
        dict_delete(dic, "a/*/b", remove_empty_ancestors=True)
        self.assertEqual(
            {"a": [{"c": 3}, {"c": 30}, {"d": 2, "c": 5}, {"c": 300}]}, dic
        )

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references")
        self.assertEqual({}, dic)

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references/*", remove_empty_ancestors=True)
        self.assertEqual({}, dic)

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references/*")
        self.assertEqual({"references": []}, dic)
        dict_delete(dic, "references")
        self.assertEqual({}, dic)

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references/")
        self.assertEqual({"references": []}, dic)

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references/1")
        self.assertEqual({"references": ["r1", "r3"]}, dic)
        dict_delete(dic, "references/8", not_exist_ok=True)
        # do nothing if not_exist_ok=True, and query asks to delete non existing path
        self.assertDictEqual({"references": ["r1", "r3"]}, dic)
        with self.assertRaises(ValueError):
            dict_delete(dic, "references/8")

        dic = {"references": ["r1", "r2", "r3"]}
        dict_delete(dic, "references", remove_empty_ancestors=True)
        self.assertEqual({}, dic)
        dic = {"references": [{"r11": 1, "r12": 2}, "r2", "r3"]}
        with self.assertRaises(ValueError):
            dict_delete(dic, "refrefs/1", remove_empty_ancestors=True)
        dict_delete(dic, "references/0/*", remove_empty_ancestors=False)
        self.assertEqual({"references": [{}, "r2", "r3"]}, dic)
        dict_delete(dic, "references/0/*", remove_empty_ancestors=True)
        self.assertEqual({"references": ["r2", "r3"]}, dic)
        with self.assertRaises(ValueError):
            dict_delete(dic, "references+#/^!", remove_empty_ancestors=True)
        dict_delete(dic, "*")
        self.assertEqual({}, dic)
        dic = {"references": [{"r11": 1, "r12": 2}, "r2", "r3"]}
        with self.assertRaises(ValueError):
            dict_delete(dic, "")
        dic = {"references": [{"r11": 1, "r12": 2}, "r2", "r3"]}
        dict_delete(dic, "*", remove_empty_ancestors=True)
        self.assertEqual({}, dic)

        with self.assertRaises(ValueError):
            dict_delete(dic, "", remove_empty_ancestors=True)

        dic = {"a": [[["i1", "i2"], ["i3", "i4"]], [["i5", "i6"], ["i7", "i8"]]]}
        dict_delete(dic, "a/1/0/1")
        self.assertEqual(
            {"a": [[["i1", "i2"], ["i3", "i4"]], [["i5"], ["i7", "i8"]]]}, dic
        )
        dict_delete(dic, "a/1/0/0")
        self.assertEqual({"a": [[["i1", "i2"], ["i3", "i4"]], [[], ["i7", "i8"]]]}, dic)
        dict_delete(dic, "a/1/1")
        self.assertEqual({"a": [[["i1", "i2"], ["i3", "i4"]], [[]]]}, dic)
        dict_delete(dic, "a/1/0", remove_empty_ancestors=True)
        self.assertEqual({"a": [[["i1", "i2"], ["i3", "i4"]]]}, dic)

        with self.assertRaises(ValueError):
            dict_delete(dic, "aaa")
        with self.assertRaises(ValueError):
            dict_delete(dic, "a/2/3")

        self.assertEqual("i", dict_get(dic, "a/0/0/0/0"))
        self.assertEqual("i", dict_get(dic, "a/0/0/0/0/0/0/0"))
        with self.assertRaises(ValueError):
            dict_get(dic, "a/0/0/0/*/0")

        dic = {
            "a": {
                "b": {"c": {"d": 7}, "g": {"d": 7}},
                "h": {"c": {"d": 7}, "g": {"d": 7}, "i": {"d": 7}},
                "w": 3,
            }
        }
        dict_delete(dic, "a/*/c")
        self.assertDictEqual(
            {"a": {"b": {"g": {"d": 7}}, "h": {"g": {"d": 7}, "i": {"d": 7}}, "w": 3}},
            dic,
        )

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
        dict_set(dic, "", {"a": 6, "b": 8})
        self.assertDictEqual({"a": 6, "b": 8}, dic)
        dic = {"x/y": 1, "b": 2}
        dict_set(dic, "x/y", 3)
        self.assertDictEqual(dic, {"x/y": 3, "b": 2})

        dic = [1, 2, 3]
        dict_set(dic, "", [])
        self.assertEqual([], dic)
        with self.assertRaises(ValueError):
            dict_set(None, "path", {"a": 6, "b": 8}, not_exist_ok=False)

    def test_nested_set(self):
        dic = {"a": {"b": 1, "c": 2}}
        dict_set(dic, "a/b", 3)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 2}})
        dict_set(dic, "a/c", 4)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 4}})
        with self.assertRaises(ValueError):
            dict_set(dic, "a/d", 5, not_exist_ok=False)
        dict_set(dic, "a/d", 5)
        self.assertDictEqual(dic, {"a": {"b": 3, "c": 4, "d": 5}})

    def test_query_set(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        dict_set(dic, "a/*/b", [3, 4], set_multiple=True)
        self.assertDictEqual(dic, {"a": [{"b": 3}, {"b": 4}]})
        dict_set(dic, "a/*/b", [3, 4], set_multiple=False)
        self.assertDictEqual(dic, {"a": [{"b": [3, 4]}, {"b": [3, 4]}]})
        dict_set(dic, "a/0/c", [])
        self.assertDictEqual({"a": [{"b": [3, 4], "c": []}, {"b": [3, 4]}]}, dic)
        dict_set(dic, "a/0/b/c/*/d", [5, 6], set_multiple=False)
        self.assertDictEqual(
            dic, {"a": [{"b": {"c": [{"d": [5, 6]}]}, "c": []}, {"b": [3, 4]}]}
        )
        # sort keys toward breakup per set_multiple
        dic = {"c": 3, "a": 1, "b": 2}
        dict_set(dic, "*", [10, 20, 30], set_multiple=True)
        self.assertDictEqual({"a": 10, "b": 20, "c": 30}, dic)

        dict_set(dic, "a/0/c/d/*/e/*/f", [7, 8], set_multiple=True)
        # breaks up just one, smoothly
        self.assertDictEqual(
            {
                "c": 30,
                "a": [{"c": {"d": [{"e": [{"f": 7}]}, {"e": [{"f": 8}]}]}}],
                "b": 20,
            },
            dic,
        )

        dict_set(dic, "a/1/c/d/*/e/*/f/*/h", [])
        self.assertDictEqual(
            {
                "c": 30,
                "a": [
                    {"c": {"d": [{"e": [{"f": 7}]}, {"e": [{"f": 8}]}]}},
                    {"c": {"d": [{"e": [{"f": [{"h": []}]}]}]}},
                ],
                "b": 20,
            },
            dic,
        )

        dic = {"c": [{"b": 3}, {"b": 4}], "a": [{"b": 1}, {"b": 2}]}
        dict_set(dic, "*/1/b", [5, 6], set_multiple=True)
        # ordered paths alphabetically before assigning
        self.assertDictEqual(
            dic, {"a": [{"b": 1}, {"b": 5}], "c": [{"b": 3}, {"b": 6}]}
        )

        with self.assertRaises(ValueError):
            # lengths do not match for dict with set_multiple, so not success, and not_exist_ok = False, so raises
            dict_set(
                dic,
                "*/1/b",
                [50, 60, 70],
                set_multiple=True,
                not_exist_ok=False,
            )

        dic = [[{"b": 1}, {"b": 5}], [{"b": 3}, {"b": 6}]]
        with self.assertRaises(ValueError):
            # list is not allowed to extend to length of value
            dict_set(
                dic,
                "*/1/b",
                [50, 60, 70],
                set_multiple=True,
                not_exist_ok=False,
            )
        with self.assertRaises(ValueError):
            # list is too long to begin with
            dict_set(
                dic,
                "*/1/b",
                [50],
                set_multiple=True,
                not_exist_ok=False,
            )

        dict_set(
            dic,
            "*/1/b",
            [50, 60, 70],
            set_multiple=True,
        )
        # list extends to the length of value
        self.assertListEqual(
            [[{"b": 1}, {"b": 50}], [{"b": 3}, {"b": 60}], [None, {"b": 70}]], dic
        )

        dic = {"a": {"b": []}}
        dict_set(dic, "a/b/2/c", [3, 4])
        self.assertDictEqual(dic, {"a": {"b": [None, None, {"c": [3, 4]}]}})

        dic = {"a": {"b": []}}
        dict_set(dic, "c", None)
        self.assertDictEqual({"a": {"b": []}, "c": None}, dic)
        dict_set(dic, "d/e/*/f/*", None)
        self.assertDictEqual(
            {"a": {"b": []}, "c": None, "d": {"e": [{"f": [None]}]}}, dic
        )
        dict_set(dic, "d/e/*/f/", None)
        self.assertDictEqual(
            {"a": {"b": []}, "c": None, "d": {"e": [{"f": [None]}]}}, dic
        )
        dict_set(dic, "d/e/*/f", None)
        self.assertDictEqual(
            {"a": {"b": []}, "c": None, "d": {"e": [{"f": None}]}}, dic
        )
        dict_set(dic, "a/b/*", 5)
        self.assertDictEqual(
            {"a": {"b": [5]}, "c": None, "d": {"e": [{"f": None}]}}, dic
        )

        dic = {"a": {"b": []}}
        with self.assertRaises(ValueError):
            dict_set(dic, "a/b/2/c", [3, 4], not_exist_ok=False)

    def test_query_set_with_multiple_non_existing(self):
        dic = {"a": [{"b": 1}, {"b": 2}]}
        dict_set(dic, "a/*/c", [3, 4], set_multiple=True)
        self.assertDictEqual({"a": [{"b": 1, "c": 3}, {"b": 2, "c": 4}]}, dic)
        dict_set(
            dic,
            "a/*/c",
            [3, 4],
            set_multiple=False,
            not_exist_ok=True,
        )
        self.assertEqual({"a": [{"b": 1, "c": [3, 4]}, {"b": 2, "c": [3, 4]}]}, dic)

        dic = {"a": [{"b": 1}, {"b": 2}]}
        dict_set(
            dic,
            "a/*/d",
            [3, 4, 5],
            set_multiple=True,
            not_exist_ok=True,
        )
        self.assertDictEqual({"a": [{"b": 1, "d": 3}, {"b": 2, "d": 4}, {"d": 5}]}, dic)

    def test_adding_one_new_field(self):
        dic = {"a": {"b": {"c": 0}}}
        dict_set(dic, "a/b/d", 1)
        self.assertDictEqual(dic, {"a": {"b": {"c": 0, "d": 1}}})

    def test_adding_one_new_field_nested(self):
        dic = {"d": 0}
        dict_set(dic, "/a/b/d", 1)
        self.assertDictEqual(dic, {"a": {"b": {"d": 1}}, "d": 0})
