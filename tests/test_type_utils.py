import unittest
import typing
from src.unitxt.type_utils import is_typing_type

class TestAssertTyping(unittest.TestCase):
    def test_simple_types(self):
        self.assertEqual(is_typing_type(1, int), True)
        self.assertEqual(is_typing_type("hello", str), True)
        self.assertEqual(is_typing_type(1.23, float), True)
        self.assertEqual(is_typing_type([1, 2, 3], typing.List[int]), True)
        self.assertEqual(is_typing_type(("hello", 1), typing.Tuple[str, int]), True)
        self.assertEqual(is_typing_type({"key": 1}, typing.Dict[str, int]), True)
        self.assertEqual(is_typing_type([1, 2, 3], typing.List[str]), False)
        self.assertEqual(is_typing_type(("hello", 1), typing.Tuple[int, int]), False)
        self.assertEqual(is_typing_type({"key": 1}, typing.Dict[int, int]), False)

    def test_nested_types(self):
        self.assertEqual(is_typing_type([[1, 2], [3, 4]], typing.List[typing.List[int]]), True)
        self.assertEqual(is_typing_type([("hello", 1), ("world", 2)], typing.List[typing.Tuple[str, int]]), True)
        self.assertEqual(is_typing_type({"key1": [1, 2], "key2": [3, 4]}, typing.Dict[str, typing.List[int]]), True)
        self.assertEqual(is_typing_type([[1, 2], [3, "4"]], typing.List[typing.List[int]]), False)
        self.assertEqual(is_typing_type([("hello", 1), ("world", "2")], typing.List[typing.Tuple[str, int]]), False)
        self.assertEqual(is_typing_type({"key1": [1, 2], "key2": [3, "4"]}, typing.Dict[str, typing.List[int]]), False)
