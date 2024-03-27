import typing

from src.unitxt.type_utils import (
    encode_type_of_obj,
    isoftype,
    issubtype,
    parse_type_string,
    to_float_or_default,
)
from tests.utils import UnitxtTestCase


class TestAssertTyping(UnitxtTestCase):
    def test_simple_types(self):
        self.assertEqual(isoftype(1, int), True)
        self.assertEqual(isoftype("hello", str), True)
        self.assertEqual(isoftype("hello", typing.List[str]), False)
        self.assertEqual(isoftype(1.23, float), True)
        self.assertEqual(isoftype([1, 2, 3], typing.List[int]), True)
        self.assertEqual(isoftype(("hello", 1), typing.Tuple[str, int]), True)
        self.assertEqual(isoftype({"key": 1}, typing.Dict[str, int]), True)
        self.assertEqual(isoftype([1, 2, 3], typing.List[str]), False)
        self.assertEqual(isoftype(("hello", 1), typing.Tuple[int, int]), False)
        self.assertEqual(isoftype({"key": 1}, typing.Dict[int, int]), False)

    def test_unions(self):
        self.assertEqual(isoftype(1, typing.Union[int, float]), True)
        self.assertEqual(isoftype(2.0, typing.Union[int, float]), True)
        self.assertEqual(isoftype("2.0", typing.Union[int, float]), False)
        self.assertEqual(isoftype(["2.0"], typing.Union[int, float]), False)
        self.assertEqual(isoftype(["2.0"], typing.Union[int, float, list]), True)
        self.assertEqual(
            isoftype(["2.0"], typing.Union[int, float, typing.List[int]]), False
        )
        self.assertEqual(
            isoftype(["2.0"], typing.Union[int, float, typing.List[str]]), True
        )
        self.assertEqual(isoftype([1], typing.Union[int, float]), False)

    def test_any(self):
        self.assertEqual(isoftype(1, typing.Any), True)
        self.assertEqual(isoftype(2.1, typing.Any), True)
        self.assertEqual(isoftype("kd", typing.Any), True)
        self.assertEqual(isoftype(["1"], typing.Any), True)
        self.assertEqual(isoftype(["1"], typing.List[typing.Any]), True)
        self.assertEqual(isoftype(["1"], typing.Tuple[typing.Any]), False)

    def test_nested_types(self):
        self.assertEqual(
            isoftype([[1, 2], [3, 4]], typing.List[typing.List[int]]), True
        )
        self.assertEqual(
            isoftype([("hello", 1), ("world", 2)], typing.List[typing.Tuple[str, int]]),
            True,
        )
        self.assertEqual(
            isoftype(
                {"key1": [1, 2], "key2": [3, 4]}, typing.Dict[str, typing.List[int]]
            ),
            True,
        )
        self.assertEqual(
            isoftype([[1, 2], [3, "4"]], typing.List[typing.List[int]]), False
        )
        self.assertEqual(
            isoftype(
                [("hello", 1), ("world", "2")], typing.List[typing.Tuple[str, int]]
            ),
            False,
        )
        self.assertEqual(
            isoftype(
                {"key1": [1, 2], "key2": [3, "4"]}, typing.Dict[str, typing.List[int]]
            ),
            False,
        )

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
        self.assertTrue(
            issubtype(typing.Dict[Name, Name2], typing.Dict[BaseName, BaseName2])
        )
        self.assertTrue(
            issubtype(typing.Tuple[Name, Name2], typing.Tuple[BaseName, BaseName2])
        )
        self.assertTrue(issubtype(Name, BaseName))
        self.assertFalse(issubtype(BaseName, Name))
        self.assertFalse(issubtype(typing.List[Name], typing.List[BaseName2]))
        self.assertFalse(
            issubtype(typing.Dict[Name, Name2], typing.Dict[BaseName, Name])
        )

    def test_to_float_or_default(self):
        self.assertEqual(to_float_or_default("1", 0), 1)
        self.assertEqual(to_float_or_default("a", 0), 0)
        with self.assertRaises(ValueError):
            to_float_or_default("a", None)

    def test_parse_basic_types(self):
        self.assertEqual(parse_type_string("int"), int)
        self.assertEqual(parse_type_string("str"), str)
        self.assertEqual(parse_type_string("float"), float)
        self.assertEqual(parse_type_string("bool"), bool)

    def test_parse_generic_types(self):
        self.assertEqual(parse_type_string("List[int]"), typing.List[int])
        self.assertEqual(parse_type_string("List[int,]"), typing.List[int])
        self.assertEqual(parse_type_string("Dict[str, int]"), typing.Dict[str, int])
        self.assertEqual(parse_type_string("Tuple[int, str]"), typing.Tuple[int, str])
        self.assertEqual(parse_type_string("Optional[str]"), typing.Optional[str])

    def test_parse_nested_generic_types(self):
        self.assertEqual(
            parse_type_string("List[Dict[str, int]]"),
            typing.List[typing.Dict[str, int]],
        )
        self.assertEqual(
            parse_type_string("Dict[str, List[int]]"),
            typing.Dict[str, typing.List[int]],
        )
        self.assertEqual(
            parse_type_string("Optional[List[str]]"), typing.Optional[typing.List[str]]
        )

    def test_encode_basic_types(self):
        self.assertEqual(encode_type_of_obj(7), "int")
        self.assertEqual(encode_type_of_obj("hello"), "str")
        self.assertEqual(encode_type_of_obj(2.5), "float")
        self.assertEqual(encode_type_of_obj(True), "bool")

    def test_enode_generic_types(self):
        self.assertEqual(encode_type_of_obj([1, 2]), "List[int]")
        self.assertEqual(encode_type_of_obj([]), "List[Any]")
        self.assertEqual(encode_type_of_obj({"how_much": 7}), "Dict[str,int]")
        self.assertEqual(encode_type_of_obj((7, "what seven")), "Tuple[int,str]")

    def test_encode_nested_generic_types(self):
        obj = ["who am I", 6, {"number 1 is": False}, []]
        self.assertTrue(isoftype(obj, parse_type_string(encode_type_of_obj(obj))))
        obj = ["who am I", 6, {"number 1 is": False}, ([], "empty", 7)]
        self.assertTrue(isoftype(obj, parse_type_string(encode_type_of_obj(obj))))
        self.assertEqual(encode_type_of_obj([{"how_much": 7}]), "List[Dict[str,int]]")
        self.assertEqual(
            encode_type_of_obj({"how many": [77, 88]}), "Dict[str,List[int]]"
        )

    def test_parse_union_type(self):
        self.assertEqual(parse_type_string("Union[int, str]"), typing.Union[int, str])
        self.assertEqual(
            parse_type_string("Union[int, List[str]]"),
            typing.Union[int, typing.List[str]],
        )

    # Adding tests designed to fail
    def test_parse_invalid_syntax(self):
        with self.assertRaises(SyntaxError):
            parse_type_string("List[int,,]")

    def test_parse_unsupported_type(self):
        with self.assertRaises(ValueError):
            parse_type_string("Set[int]")

    def test_parse_malformed_string(self):
        with self.assertRaises(TypeError):
            parse_type_string("List[[int]]")
