import typing

from unitxt.type_utils import (
    format_type_string,
    infer_type,
    infer_type_string,
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

    def test_infer_type_string_basics(self):
        self.assertEqual(infer_type_string(7), "int")
        self.assertEqual(infer_type_string("hello"), "str")
        self.assertEqual(infer_type_string(2.5), "float")
        self.assertEqual(infer_type_string(True), "bool")

        class Cl:
            pass

        obj = Cl()
        self.assertEqual("Any", infer_type_string(obj))

    def test_infer_type_string_generic_types(self):
        self.assertEqual(infer_type_string([1, 2]), "List[int]")
        self.assertEqual(infer_type_string([]), "List[Any]")
        self.assertEqual(infer_type_string([[], [7]]), "List[List[int]]")
        self.assertEqual(
            infer_type_string([[], [7], ["seven"]]), "List[Union[List[int],List[str]]]"
        )
        self.assertEqual(
            infer_type_string([[], 7, "seven"]), "List[Union[List[Any],int,str]]"
        )
        self.assertEqual(
            infer_type_string([{}, 7, True]), "List[Union[Dict[Any,Any],int]]"
        )
        self.assertEqual(infer_type_string({"how_much": 7}), "Dict[str,int]")
        self.assertEqual(infer_type_string((7, "what seven")), "Tuple[int,str]")
        self.assertEqual(
            infer_type_string([(2, 3, "four"), (6, 7, "eight")]),
            "List[Tuple[int,int,str]]",
        )
        obj = [(2, 3, "four"), (6, 7, "eight", 9)]
        # tuples of different length do not go into same type
        self.assertEqual(
            "List[Union[Tuple[int,int,str,int],Tuple[int,int,str]]]",
            infer_type_string(obj),
        )

    def test_to_string_nested_generic_types(self):
        obj = []
        # no type specified for the string element.
        self.assertEqual("List[Any]", infer_type_string(obj))
        obj = [[]]
        # no type specified for the string element.
        self.assertEqual("List[List[Any]]", infer_type_string(obj))
        obj = [[], [8, 9]]
        # one of the sublists does specifies a type, and this applies to its sister sublist.
        self.assertEqual("List[List[int]]", infer_type_string(obj))
        obj = ["who am I", 6, {"number 1 is": False}, []]
        self.assertTrue(isoftype(obj, infer_type(obj)))
        obj = [
            ["who am I", 6, {"number 1 is": False}, []],
            [{"empty": [], "number 1 is": [1]}, [99], ["hello"]],
        ]
        self.assertTrue(isoftype(obj, infer_type(obj)))
        obj = ["who am I", 6, {"number 1 is": False}, ([], "empty", 7)]
        self.assertTrue(isoftype(obj, infer_type(obj)))
        self.assertEqual(infer_type_string([{"how_much": 7}]), "List[Dict[str,int]]")
        self.assertEqual(
            infer_type_string({"how many": [77, 88]}), "Dict[str,List[int]]"
        )
        obj = [
            ["who am I", 6, {"number 1 is": False}, [], {"number 2 is": 2}, {}],
            [
                {"empty": [], "number 1 is": [1]},
                [99],
                ["hello"],
                (2, 4, 6),
                (1, 3, 5),
                ("e", "f", "g"),
                (),
            ],
        ]
        self.assertEqual(
            "List[Union[List[Union[Dict[str,List[int]],List[int],List[str],Tuple[Any],Tuple[int,int,int],Tuple[str,str,str]]],List[Union[Dict[str,int],List[Any],int,str]]]]",
            infer_type_string(obj),
        )
        obj = [
            [{"number 1 is": False}, [], {"number 2 is": 2}, {}, ()],
            [
                {"empty": [], "number 1 is": [1]},
                (2, 4, 6),
                (1, 3, True),
                ("e", "f", "g"),
                (),
            ],
        ]
        self.assertEqual(
            "List[Union[List[Union[Dict[str,List[int]],Tuple[Any],Tuple[int,int,int],Tuple[str,str,str]]],List[Union[Dict[str,int],List[Any],Tuple[Any]]]]]",
            infer_type_string(obj),
        )

        obj = [
            {"number 1 is": False},
            [],
            {"number 2 is": 2},
            {},
            (),
            {"number 1 is": [1]},
            (1, 2, 3),
        ]
        self.assertEqual(
            "List[Union[Dict[str,List[int]],Dict[str,int],List[Any],Tuple[Any],Tuple[int,int,int]]]",
            infer_type_string(obj),
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

    def test_format_type_string(self):
        self.assertEqual(
            format_type_string(
                "typing.Tuple[int,float,int|list[int|dict[str,int|float]]]"
            ),
            "Tuple[int,float,Union[int,List[Union[int,Dict[str,Union[int,float]]]]]]",
        )
        self.assertEqual(
            parse_type_string("tuple[int, str, typing.List[int | float]]"),
            typing.Tuple[int, str, typing.List[typing.Union[int, float]]],
        )
