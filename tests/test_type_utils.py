import unittest
import typing
from src.unitxt.type_utils import isoftype, issubtype


class TestAssertTyping(unittest.TestCase):
    def test_simple_types(self):
        self.assertEqual(isoftype(1, int), True)
        self.assertEqual(isoftype("hello", str), True)
        self.assertEqual(isoftype(1.23, float), True)
        self.assertEqual(isoftype([1, 2, 3], typing.List[int]), True)
        self.assertEqual(isoftype(("hello", 1), typing.Tuple[str, int]), True)
        self.assertEqual(isoftype({"key": 1}, typing.Dict[str, int]), True)
        self.assertEqual(isoftype([1, 2, 3], typing.List[str]), False)
        self.assertEqual(isoftype(("hello", 1), typing.Tuple[int, int]), False)
        self.assertEqual(isoftype({"key": 1}, typing.Dict[int, int]), False)

    def test_nested_types(self):
        self.assertEqual(isoftype([[1, 2], [3, 4]], typing.List[typing.List[int]]), True)
        self.assertEqual(isoftype([("hello", 1), ("world", 2)], typing.List[typing.Tuple[str, int]]), True)
        self.assertEqual(isoftype({"key1": [1, 2], "key2": [3, 4]}, typing.Dict[str, typing.List[int]]), True)
        self.assertEqual(isoftype([[1, 2], [3, "4"]], typing.List[typing.List[int]]), False)
        self.assertEqual(isoftype([("hello", 1), ("world", "2")], typing.List[typing.Tuple[str, int]]), False)
        self.assertEqual(isoftype({"key1": [1, 2], "key2": [3, "4"]}, typing.Dict[str, typing.List[int]]), False)

    def test_is_typing_sub_type(self):
        # Define some base classes and subclasses
        class BaseName:
            pass

        class Name(BaseName):
            pass

        class BaseName2:
            pass

        class Name2(BaseName2):
            pass

        self.assertTrue(issubtype(typing.List[Name], typing.List[BaseName]))
        self.assertTrue(issubtype(typing.Dict[Name, Name2], typing.Dict[BaseName, BaseName2]))
        self.assertTrue(issubtype(typing.Tuple[Name, Name2], typing.Tuple[BaseName, BaseName2]))
        self.assertTrue(issubtype(Name, BaseName))
        self.assertFalse(issubtype(BaseName, Name))
        self.assertFalse(issubtype(typing.List[Name], typing.List[BaseName2]))
        self.assertFalse(issubtype(typing.Dict[Name, Name2], typing.Dict[BaseName, Name]))
