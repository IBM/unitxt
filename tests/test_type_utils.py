import collections.abc
import io
import typing
import unittest

from src.unitxt.type_utils import (
    NormalizedType,
    get_args,
    get_origin,
    isoftype,
    issubtype,
    normalize,
    unknown,
)

JSON = typing.Union[
    int, float, bool, str, None, typing.Sequence["JSON"], typing.Mapping[str, "JSON"]
]


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

    def test_normalize(self):
        # None
        assert normalize(None) == normalize(type(None))

        # basic types
        assert normalize(list) == normalize(typing.List) == list

        # abstract types
        assert normalize(collections.abc.Sequence) == normalize(typing.Sequence)

        # common generic types
        assert normalize(typing.Union) == typing.Union
        assert (
            normalize(typing.Union[int, typing.List, list])
            == normalize(typing.Union[list, int])
            == NormalizedType(typing.Union, frozenset((list, int)))
        )
        assert normalize(typing.Union[typing.List, list]) != normalize(
            typing.Union[typing.Sequence, int]
        )

        # Union
        assert normalize(typing.Union[typing.List[int], int]) == normalize(
            typing.Union[int, typing.List[int], int]
        )
        assert normalize(typing.Union[typing.List[int], int]) != normalize(
            typing.Union[typing.List, int]
        )

        # collections
        assert normalize(typing.List) == normalize(list) == list

        # Callable
        assert (
            normalize(typing.Callable[[typing.List, int], None])
            == normalize(typing.Callable[[list, int], None])
            == NormalizedType(collections.abc.Callable, ((list, int), None))
        )
        assert normalize(typing.Callable[[typing.List, int], None]) != normalize(
            typing.Callable[[int, list], None]
        )

    def test_generic_utils(self):
        assert get_origin(list) is None
        assert get_origin(typing.Union) is None

        assert get_args(typing.List) == tuple()
        assert get_origin(typing.List) == list

        assert get_args(typing.List[int]) == (int,)
        assert get_origin(typing.List[int]) == list
        assert get_args(typing.List[str]) != (int,)

        assert get_origin(typing.Union[int, str]) == typing.Union
        assert get_args(typing.Union[int, str]) == (int, str)

        fun = typing.Callable[[str, int], int]
        assert get_origin(fun) == collections.abc.Callable
        assert get_args(fun) == ([str, int], int)

        PayloadType = typing.TypeVar("PayloadType")

        class TypeA(typing.Generic[PayloadType]):
            def __init__(self, payload: PayloadType):
                self.payload = payload

        assert get_origin(TypeA[int]) == TypeA
        assert get_args(TypeA[int]) == (int,)

    def test_is_subtype(self):
        # Any
        assert issubtype(typing.List, typing.Any)
        assert issubtype(typing.Any, typing.Any)

        # Self
        assert issubtype(list, list)
        assert issubtype(typing.List, typing.List)
        assert not issubtype(list, dict)
        assert not issubtype(typing.List, typing.Dict)

        # None
        assert issubtype(None, type(None))
        assert issubtype(type(None), None)
        assert issubtype(None, None)

        # alias
        assert issubtype(list, typing.List)
        assert issubtype(typing.List, list)
        assert issubtype(bytes, typing.ByteString)

        # Subclass
        assert issubtype(list, typing.Sequence)

        # FileLike
        with open("test", "wb") as file_ref:
            assert issubtype(type(file_ref), typing.BinaryIO)
        with open("test", "rb") as file_ref:
            assert issubtype(type(file_ref), typing.BinaryIO)
        with open("test", "w") as file_ref:
            assert issubtype(type(file_ref), typing.TextIO)
        with open("test", "r") as file_ref:
            assert issubtype(type(file_ref), typing.TextIO)

        assert issubtype(type(io.BytesIO(b"0")), typing.BinaryIO)
        assert issubtype(type(io.StringIO("0")), typing.TextIO)

        # subscribed generic
        assert issubtype(typing.List[int], list)
        assert issubtype(typing.List[typing.List], list)
        assert not issubtype(list, typing.List[int])

        # Union
        assert issubtype(list, typing.Union[typing.List, typing.Tuple])
        assert issubtype(typing.Union[list, tuple], typing.Union[list, tuple, None])
        assert issubtype(typing.Union[list, tuple], typing.Sequence)

        assert not issubtype(list, typing.Union[typing.Tuple, typing.Set])
        assert not issubtype(typing.Tuple[typing.Union[int, None]], typing.Tuple[None])

        # Nested containers
        assert issubtype(typing.List[int], typing.List[int])
        assert issubtype(typing.List[typing.List], typing.List[typing.Sequence])

        assert issubtype(
            typing.Dict[typing.List, int], typing.Dict[typing.Sequence, int]
        )
        assert issubtype(
            typing.Callable[[typing.List, int], int],
            typing.Callable[[typing.Sequence, int], int],
        )
        assert not issubtype(
            typing.Callable[[typing.Sequence, int], int],
            typing.Callable[[typing.List, int], int],
        )

        # Callable
        assert issubtype(
            typing.Callable[[typing.List, int], None],
            typing.Callable[[list, int], None],
        )
        assert issubtype(
            typing.Callable[[typing.List, int], None],
            typing.Callable[[typing.Sequence, int], None],
        )
        assert issubtype(
            typing.Callable[[typing.List[int], int], None],
            typing.Callable[[typing.Sequence[int], int], None],
        )
        assert not issubtype(
            typing.Callable[[typing.List[int], int], None],
            typing.Callable[[typing.List[str], int], None],
        )
        assert not issubtype(
            typing.Callable[[typing.List[int], int], None],
            typing.Callable[[typing.List[int], int], int],
        )
        assert not issubtype(
            typing.Callable[[typing.List[int], int, None], None],
            typing.Callable[[typing.List[int], int], None],
        )

        # ForwardRef
        assert issubtype(int, JSON, forward_refs={"JSON": JSON})
        assert issubtype(str, JSON, forward_refs={"JSON": JSON})
        assert issubtype(typing.Dict[str, str], JSON, forward_refs={"JSON": JSON})
        assert not issubtype(typing.Dict[str, bytes], JSON, forward_refs={"JSON": JSON})

        assert issubtype(
            typing.Dict[str, str],
            typing.Union[JSON, bytes],
            forward_refs={"JSON": JSON},
        )
        assert not issubtype(
            typing.Dict[str, bytes],
            typing.Union[JSON, bytes],
            forward_refs={"JSON": JSON},
        )

        # Ellipsis
        assert issubtype(typing.Tuple[list], typing.Tuple[list, ...])
        assert issubtype(typing.Tuple[typing.List], typing.Tuple[list, ...])
        assert issubtype(typing.Tuple[list, list], typing.Tuple[typing.Sequence, ...])
        assert not issubtype(
            typing.Tuple[list, int], typing.Tuple[typing.Sequence, ...]
        )
        assert issubtype(typing.Tuple[list, ...], typing.Tuple[list, ...])
        assert issubtype(typing.Tuple[list, ...], typing.Tuple[typing.Sequence, ...])
        assert not issubtype(typing.Tuple[list, ...], typing.Tuple[list])

        # TypeVar
        T1 = typing.TypeVar("T1")
        T2 = typing.TypeVar("T2")
        T3 = typing.TypeVar("T3", bound=str)
        T4 = typing.TypeVar("T4", bound="typing.Union[list, tuple]")
        assert issubtype(T1, T1)
        assert not issubtype(T1, T2) and issubtype(T1, T2) is unknown
        assert not issubtype(T3, T4) and issubtype(T3, T4) is not unknown
        assert issubtype(T3, str)
        assert issubtype(T4, typing.Sequence)
