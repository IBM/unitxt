import typing
from typing import Literal, NewType, TypedDict

from unitxt.type_utils import (
    UnsupportedTypeError,
    format_type_string,
    infer_type,
    infer_type_string,
    is_type,
    isoftype,
    issubtype,
    parse_type_string,
    register_type,
    replace_class_names,
    to_float_or_default,
    to_type_string,
    verify_required_schema,
)

from tests.utils import UnitxtTestCase


class TestAssertTyping(UnitxtTestCase):
    def test_new_type(self):
        UserId = NewType("UserId", int)
        self.assertEqual(isoftype(UserId(1), UserId), True)
        self.assertEqual(isoftype(1, UserId), True)  # Since UserId is based on int
        self.assertEqual(isoftype("1", UserId), False)

    def test_typed_dict(self):
        class Person(TypedDict):
            name: str
            age: int

        person: Person = {"name": "Alice", "age": 30}
        self.assertEqual(isoftype(person, Person), True)

        invalid_person = {
            "name": "Alice",
            "age": "30",
        }  # Age is a string, should be int
        self.assertEqual(isoftype(invalid_person, Person), False)

        incomplete_person = {"name": "Alice"}  # Missing age
        self.assertEqual(isoftype(incomplete_person, Person), False)

    def test_literal(self):
        valid_literal: Literal[1, 2, 3] = 2
        invalid_literal: Literal[1, 2, 3] = 4  # 4 is not part of the Literal

        self.assertEqual(isoftype(valid_literal, Literal[1, 2, 3]), True)
        self.assertEqual(isoftype(invalid_literal, Literal[1, 2, 3]), False)

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

    def test_handling_registered_types(self):
        UserId = NewType("UserId", int)
        register_type(UserId)
        self.assertEqual(parse_type_string("UserId"), UserId)
        self.assertEqual(parse_type_string("List[UserId]"), typing.List[UserId])
        self.assertEqual(to_type_string(typing.List[UserId]), "List[UserId]")

        class Person(TypedDict):
            name: str
            age: int

        register_type(Person)
        self.assertEqual(parse_type_string("Person"), Person)
        self.assertEqual(parse_type_string("Tuple[Person]"), typing.Tuple[Person])
        self.assertEqual(to_type_string(typing.Tuple[Person]), "Tuple[Person]")

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

    def test_parse_type_string_with_literal(self):
        self.assertEqual(parse_type_string("Literal['3', 3]"), typing.Literal["3", 3])
        self.assertEqual(
            parse_type_string("Union[List[str], List[Literal['3', 3]]]"),
            typing.Union[typing.List[str], typing.List[typing.Literal["3", 3]]],
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

    def test_verify_required_schema(self):
        schema = {
            "field_1": typing.Dict[str, float],
            "field_2": int,
            "field_3": typing.Tuple[typing.List[str], typing.Optional[str]],
        }

        obj = {
            "field_1": {"a": 5.0, "b": 0.5},
            "field_2": 1,
            "field_3": (["a", "b"], None),
        }
        verify_required_schema(
            schema, obj, class_name="Task", id="my_task", description="This is my task."
        )

        obj_2 = obj.copy()
        obj_2.update({"field_1": {"a": "b"}})
        with self.assertRaises(Exception) as e:
            verify_required_schema(
                schema,
                obj_2,
                class_name="Task",
                id="my_task",
                description="This is my task.",
            )
        self.assertEqual(
            str(e.exception),
            """Passed value '{'a': 'b'}' of field 'field_1' is not of required type: (Dict[str, float]) in Task ('my_task').
Task description: This is my task.""",
        )

        obj_3 = obj.copy()
        obj_3.pop("field_2")
        with self.assertRaises(Exception) as e:
            verify_required_schema(
                schema,
                obj_3,
                class_name="Task",
                id="my_task",
                description="This is my task.",
            )
        self.assertEqual(
            str(e.exception).strip('"'),
            """The Task ('my_task') expected a field 'field_2' which the input instance did not contain.
The input instance fields are  : ['field_1', 'field_3'].
Task description: This is my task.""",
        )

    def test_format_type_string(self):
        self.assertEqual(
            "Tuple[int,float,Union[int,List[Union[int,Dict[str,Union[int,float]]]]]]",
            format_type_string(
                "typing.Tuple[int,float,int|list[int|dict[str,int|float]]]"
            ),
        )

        self.assertEqual(
            "Optional[Union[int,float,bool]]",
            format_type_string("typing.Optional[int|float|bool]"),
        )

        self.assertEqual(
            "Tuple[Union[int,float,bool,List[Union[str,bool]]],Dict[str,Tuple[Union[bool,[str]]]],[[[int]]]]",
            format_type_string(
                "typing.tuple[int|float|bool|list[str|bool], dict[str, tuple[bool|[str]]], [[[int]]]]"
            ),
        )
        self.assertEqual(
            'Tuple[Union[int,float,Literal["lef[,t","rig],ht"],bool,List[Union[str,bool]]],Dict[str,Tuple[Union[bool,[str]]]],[[[int]]]]',
            format_type_string(
                'typing.tuple[int|float|Literal["lef[|t", "rig],ht"]| bool|list[str|bool], dict[str, tuple[bool|[str]]], [[[int]]]]'
            ),
        )
        self.assertEqual(
            "Tuple[int,Union[float,bool,str],Union[int,List[Union[int,Dict[str,Union[int,float]]]]]]",
            format_type_string(
                "typing.Tuple[int,float|bool|str,int|list[int|dict[str,int|float]]]"
            ),
        )

        self.assertEqual(
            parse_type_string("tuple[int | str | typing.List[int | float]]"),
            typing.Tuple[typing.Union[int, str, typing.List[typing.Union[int, float]]]],
        )

        self.assertEqual("Union[int,List[int]]", format_type_string("int|List[int]"))

        self.assertEqual(
            "Union[List[Union[int,float]],Tuple[Union[int,float]]]",
            format_type_string("List[int|float]|Tuple[int|float]"),
        )

    def test_is_type(self):
        self.assertTrue(is_type(typing.Dict[str, str]))
        self.assertTrue(is_type(typing.List[str]))
        self.assertTrue(is_type(typing.Tuple[str, str]))
        self.assertTrue(is_type(typing.Union[str, int]))
        self.assertTrue(is_type(typing.Optional[str]))
        self.assertTrue(is_type(str))
        self.assertTrue(is_type(float))
        self.assertTrue(is_type(int))
        self.assertTrue(is_type(list))
        self.assertTrue(is_type(dict))
        self.assertTrue(is_type(Literal[1, 2, 3]))
        self.assertFalse(is_type([1, 2]))
        self.assertFalse(is_type(print))

        with self.assertRaises(UnsupportedTypeError):
            isoftype(4, (int, int))

        with self.assertRaises(UnsupportedTypeError):
            isoftype(3, "int")

        with self.assertRaises(UnsupportedTypeError):
            isoftype(3, typing.List)

    def test_replace_class_names(self):
        test_cases = [
            # Basic case with <locals>
            {
                "input": "library.test_type_utils.TestAssertTyping.test_parse_registered_types.<locals>.Person",
                "expected": "Person",
            },
            # Basic case without <locals>
            {
                "input": "library.test_type_utils.TestAssertTyping.test_parse_registered_types.Person",
                "expected": "Person",
            },
            # Tuple with <locals>
            {
                "input": "Tuple[library.test_type_utils.TestAssertTyping.test_parse_registered_types.<locals>.Person]",
                "expected": "Tuple[Person]",
            },
            # Tuple without <locals>
            {
                "input": "Tuple[library.test_type_utils.TestAssertTyping.test_parse_registered_types.Person]",
                "expected": "Tuple[Person]",
            },
            # Nested structure with <locals>
            {
                "input": "List[Dict[str, List[library.test_type_utils.<locals>.Person]]]",
                "expected": "List[Dict[str, List[Person]]]",
            },
            # Multiple classes with <locals>
            {
                "input": "Tuple[library.test_type_utils.TestAssertTyping.Person, List[library.other_module.<locals>.Car]]",
                "expected": "Tuple[Person, List[Car]]",
            },
            # No replacement needed
            {"input": "Person", "expected": "Person"},
            # No match
            {"input": "Tuple[SomethingElse]", "expected": "Tuple[SomethingElse]"},
        ]

        for case in test_cases:
            with self.subTest(case=case):
                self.assertEqual(replace_class_names(case["input"]), case["expected"])


class TestToTypeString(UnitxtTestCase):
    def test_basic_types(self):
        self.assertEqual(to_type_string(int), "int")
        self.assertEqual(to_type_string(str), "str")
        self.assertEqual(to_type_string(float), "float")
        self.assertEqual(to_type_string(typing.Any), "Any")

    def test_union_type(self):
        self.assertEqual(to_type_string(typing.Union[int, str]), "Union[int, str]")

    def test_list_type(self):
        self.assertEqual(to_type_string(typing.List[int]), "List[int]")

    def test_dict_type(self):
        self.assertEqual(to_type_string(typing.Dict[str, int]), "Dict[str, int]")

    def test_optional_type(self):
        self.assertEqual(to_type_string(typing.Optional[int]), "Optional[int]")

    def test_tuple_type(self):
        self.assertEqual(to_type_string(typing.Tuple[int, str]), "Tuple[int, str]")

    def test_literal_type(self):
        self.assertEqual(to_type_string(Literal[1, 2, 3]), "Literal[1, 2, 3]")

    def test_newtype(self):
        UserId = NewType("UserId", int)
        register_type(UserId)
        self.assertEqual(to_type_string(UserId), "UserId")
        self.assertEqual(
            to_type_string(typing.Tuple[int, typing.Tuple[int, UserId]]),
            "Tuple[int, Tuple[int, UserId]]",
        )

    def test_typed_dict(self):
        class Point(TypedDict):
            x: int
            y: int

        register_type(Point)
        self.assertEqual(to_type_string(Point), "Point")

    def test_invalid_type(self):
        with self.assertRaises(ValueError):
            to_type_string(object)
