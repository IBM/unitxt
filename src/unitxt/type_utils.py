import collections.abc
import io
import itertools
import typing


def isoftype(object, type):
    """Checks if an object is of a certain typing type, including nested types.

    This function supports simple types (like `int`, `str`), typing types
    (like `List[int]`, `Tuple[str, int]`, `Dict[str, int]`), and nested typing
    types (like `List[List[int]]`, `Tuple[List[str], int]`, `Dict[str, List[int]]`).

    Args:
        object: The object to check.
        type: The typing type to check against.

    Returns:
        bool: True if the object is of the specified type, False otherwise.

    Examples:
    .. highlight:: python
    .. code-block:: python

        isoftype(1, int) # True
        isoftype([1, 2, 3], typing.List[int]) # True
        isoftype([1, 2, 3], typing.List[str]) # False
        isoftype([[1, 2], [3, 4]], typing.List[typing.List[int]]) # True
    """
    if type == typing.Any:
        return True

    if hasattr(type, "__origin__"):
        origin = type.__origin__
        type_args = typing.get_args(type)

        if origin is typing.Union:
            return any(isoftype(object, sub_type) for sub_type in type_args)

        if not isinstance(object, origin):
            return False

        if origin is list or origin is set:
            return all(isoftype(element, type_args[0]) for element in object)

        if origin is dict:
            return all(
                isoftype(key, type_args[0]) and isoftype(value, type_args[1])
                for key, value in object.items()
            )
        if origin is tuple:
            return all(
                isoftype(element, type_arg)
                for element, type_arg in zip(object, type_args)
            )
        return None

    return isinstance(object, type)


# copied from: https://github.com/bojiang/typing_utils/blob/main/typing_utils/__init__.py
# liscened under Apache License 2.0


if hasattr(typing, "ForwardRef"):  # python3.8
    ForwardRef = typing.ForwardRef
elif hasattr(typing, "_ForwardRef"):  # python3.6
    ForwardRef = typing._ForwardRef
else:
    raise NotImplementedError()


unknown = None


BUILTINS_MAPPING = {
    typing.List: list,
    typing.Set: set,
    typing.Dict: dict,
    typing.Tuple: tuple,
    typing.ByteString: bytes,  # https://docs.python.org/3/library/typing.html#typing.ByteString
    typing.Callable: collections.abc.Callable,
    typing.Sequence: collections.abc.Sequence,
    type(None): None,
}


STATIC_SUBTYPE_MAPPING: typing.Dict[type, typing.Type] = {
    io.TextIOWrapper: typing.TextIO,
    io.TextIOBase: typing.TextIO,
    io.StringIO: typing.TextIO,
    io.BufferedReader: typing.BinaryIO,
    io.BufferedWriter: typing.BinaryIO,
    io.BytesIO: typing.BinaryIO,
}


def optional_all(elements) -> typing.Optional[bool]:
    if all(elements):
        return True
    if all(e is False for e in elements):
        return False
    return unknown


def optional_any(elements) -> typing.Optional[bool]:
    if any(elements):
        return True
    if any(e is None for e in elements):
        return unknown
    return False


def _hashable(value):
    """Determine whether `value` can be hashed."""
    try:
        hash(value)
    except TypeError:
        return False
    return True


get_type_hints = typing.get_type_hints

GenericClass = type(typing.List)
UnionClass = type(typing.Union)

Type = typing.Union[None, type, "typing.TypeVar"]
OriginType = typing.Union[None, type]
TypeArgs = typing.Union[type, typing.AbstractSet[type], typing.Sequence[type]]


def _normalize_aliases(type_: Type) -> Type:
    if isinstance(type_, typing.TypeVar):
        return type_

    assert _hashable(type_), "_normalize_aliases should only be called on element types"

    if type_ in BUILTINS_MAPPING:
        return BUILTINS_MAPPING[type_]
    return type_


def get_origin(type_):
    """Get the unsubscripted version of a type.

    This supports generic types, Callable, Tuple, Union, Literal, Final and ClassVar.
    Return None for unsupported types.

    Examples:
        Here are some code examples using `get_origin` from the `typing_utils` module:

        .. code-block:: python

            from typing_utils import get_origin

            # Examples of get_origin usage
            get_origin(Literal[42]) is Literal  # True
            get_origin(int) is None  # True
            get_origin(ClassVar[int]) is ClassVar  # True
            get_origin(Generic) is Generic  # True
            get_origin(Generic[T]) is Generic  # True
            get_origin(Union[T, int]) is Union  # True
            get_origin(List[Tuple[T, T]][int]) == list  # True

    """
    if hasattr(typing, "get_origin"):  # python 3.8+
        _getter = typing.get_origin
        ori = _getter(type_)
    elif hasattr(typing.List, "_special"):  # python 3.7
        if isinstance(type_, GenericClass) and not type_._special:
            ori = type_.__origin__
        elif hasattr(type_, "_special") and type_._special:
            ori = type_
        elif type_ is typing.Generic:
            ori = typing.Generic
        else:
            ori = None
    else:  # python 3.6
        if isinstance(type_, GenericClass):
            ori = type_.__origin__
            if ori is None:
                ori = type_
        elif isinstance(type_, UnionClass):
            ori = type_.__origin__
        elif type_ is typing.Generic:
            ori = typing.Generic
        else:
            ori = None
    return _normalize_aliases(ori)


def get_args(type_) -> typing.Tuple:
    """Get type arguments with all substitutions performed.

    For unions, basic simplifications used by Union constructor are performed.

    Examples:
        Here are some code examples using `get_args` from the `typing_utils` module:

        .. code-block:: python

            from typing_utils import get_args

            # Examples of get_args usage
            get_args(Dict[str, int]) == (str, int)  # True
            get_args(int) == ()  # True
            get_args(Union[int, Union[T, int], str][int]) == (int, str)  # True
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])  # True
            get_args(Callable[[], T][int]) == ([], int)  # True
    """
    if hasattr(typing, "get_args"):  # python 3.8+
        _getter = typing.get_args
        res = _getter(type_)
    elif hasattr(typing.List, "_special"):  # python 3.7
        if (
            isinstance(type_, GenericClass) and not type_._special
        ):  # backport for python 3.8
            res = type_.__args__
            if get_origin(type_) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
        else:
            res = ()
    else:  # python 3.6
        if isinstance(type_, (GenericClass, UnionClass)):  # backport for python 3.8
            res = type_.__args__
            if get_origin(type_) is collections.abc.Callable and res[0] is not Ellipsis:
                res = (list(res[:-1]), res[-1])
        else:
            res = ()
    return () if res is None else res


def eval_forward_ref(ref, forward_refs=None):
    """Eval forward_refs in all cPython versions."""
    localns = forward_refs or {}

    if hasattr(typing, "_eval_type"):  # python3.8 & python 3.9
        _eval_type = typing._eval_type
        return _eval_type(ref, globals(), localns)

    if hasattr(ref, "_eval_type"):  # python3.6
        _eval_type = ref._eval_type
        return _eval_type(globals(), localns)

    raise NotImplementedError()


class NormalizedType(typing.NamedTuple):
    """Normalized type, made it possible to compare, hash between types."""

    origin: Type
    args: typing.Union[tuple, frozenset] = ()

    def __eq__(self, other):
        if isinstance(other, NormalizedType):
            if self.origin != other.origin:
                return False
            if isinstance(self.args, frozenset) and isinstance(other.args, frozenset):
                return self.args <= other.args and other.args <= self.args
            return self.origin == other.origin and self.args == other.args
        if not self.args:
            return self.origin == other
        return False

    def __hash__(self) -> int:
        if not self.args:
            return hash(self.origin)
        return hash((self.origin, self.args))

    def __repr__(self):
        if not self.args:
            return f"{self.origin}"
        return f"{self.origin}[{self.args}])"


def _normalize_args(tps: TypeArgs):
    if isinstance(tps, str):
        return tps
    if isinstance(tps, collections.abc.Sequence):
        return tuple(_normalize_args(type_) for type_ in tps)
    if isinstance(tps, collections.abc.Set):
        return frozenset(_normalize_args(type_) for type_ in tps)
    return normalize(tps)


def normalize(type_: Type) -> NormalizedType:
    """Convert types to NormalizedType instances."""
    args = get_args(type_)
    origin = get_origin(type_)
    if not origin:
        return NormalizedType(_normalize_aliases(type_))
    origin = _normalize_aliases(origin)

    if origin is typing.Union:  # sort args when the origin is Union
        args = _normalize_args(frozenset(args))
    else:
        args = _normalize_args(args)
    return NormalizedType(origin, args)


def _is_origin_subtype(left: OriginType, right: OriginType) -> bool:
    if left is right:
        return True

    if (
        left is not None
        and left in STATIC_SUBTYPE_MAPPING
        and right == STATIC_SUBTYPE_MAPPING[left]
    ):
        return True

    if hasattr(left, "mro"):
        for parent in left.mro():
            if parent == right:
                return True

    if isinstance(left, type) and isinstance(right, type):
        return issubclass(left, right)

    return left == right


NormalizedTypeArgs = typing.Union[
    typing.Tuple["NormalizedTypeArgs", ...],
    typing.FrozenSet[NormalizedType],
    NormalizedType,
]


def _is_origin_subtype_args(
    left: NormalizedTypeArgs,
    right: NormalizedTypeArgs,
    forward_refs: typing.Optional[typing.Mapping[str, type]],
) -> typing.Optional[bool]:
    if isinstance(left, frozenset):
        if not isinstance(right, frozenset):
            return False

        excluded = left - right
        if not excluded:
            # Union[str, int] <> Union[int, str]
            return True

        # Union[list, int] <> Union[typing.Sequence, int]
        return all(
            any(_is_normal_subtype(e, r, forward_refs) for r in right) for e in excluded
        )

    if isinstance(left, collections.abc.Sequence) and not isinstance(
        left, NormalizedType
    ):
        if not isinstance(right, collections.abc.Sequence) or isinstance(
            right, NormalizedType
        ):
            return False

        if (
            left
            and left[-1].origin is not Ellipsis
            and right
            and right[-1].origin is Ellipsis
        ):
            # Tuple[type, type] <> Tuple[type, ...]
            return all(
                _is_origin_subtype_args(lft, right[0], forward_refs) for lft in left
            )

        if len(left) != len(right):
            return False

        return all(
            lft is not None
            and rgt is not None
            and _is_origin_subtype_args(lft, rgt, forward_refs)
            for lft, rgt in itertools.zip_longest(left, right)
        )

    assert isinstance(left, NormalizedType)
    assert isinstance(right, NormalizedType)

    return _is_normal_subtype(left, right, forward_refs)


def _is_normal_subtype(
    left: NormalizedType,
    right: NormalizedType,
    forward_refs: typing.Optional[typing.Mapping[str, type]],
) -> typing.Optional[bool]:
    if isinstance(left.origin, ForwardRef):
        left = normalize(eval_forward_ref(left.origin, forward_refs=forward_refs))

    if isinstance(right.origin, ForwardRef):
        right = normalize(eval_forward_ref(right.origin, forward_refs=forward_refs))

    # Any
    if right.origin is typing.Any:
        return True

    # Union
    if right.origin is typing.Union and left.origin is typing.Union:
        return _is_origin_subtype_args(left.args, right.args, forward_refs)
    if right.origin is typing.Union:
        return optional_any(
            _is_normal_subtype(left, a, forward_refs) for a in right.args
        )
    if left.origin is typing.Union:
        return optional_all(
            _is_normal_subtype(a, right, forward_refs) for a in left.args
        )

    # TypeVar
    if isinstance(left.origin, typing.TypeVar) and isinstance(
        right.origin, typing.TypeVar
    ):
        if left.origin is right.origin:
            return True

        left_bound = getattr(left.origin, "__bound__", None)
        right_bound = getattr(right.origin, "__bound__", None)
        if right_bound is None or left_bound is None:
            return unknown
        return _is_normal_subtype(
            normalize(left_bound), normalize(right_bound), forward_refs
        )
    if isinstance(right.origin, typing.TypeVar):
        return unknown
    if isinstance(left.origin, typing.TypeVar):
        left_bound = getattr(left.origin, "__bound__", None)
        if left_bound is None:
            return unknown
        return _is_normal_subtype(normalize(left_bound), right, forward_refs)

    if not left.args and not right.args:
        return _is_origin_subtype(left.origin, right.origin)

    if not right.args:
        return _is_origin_subtype(left.origin, right.origin)

    if _is_origin_subtype(left.origin, right.origin):
        return _is_origin_subtype_args(left.args, right.args, forward_refs)

    return False


def issubtype(
    left: Type,
    right: Type,
    forward_refs: typing.Optional[dict] = None,
) -> typing.Optional[bool]:
    """Check that the left argument is a subtype of the right.

    For unions, check if the type arguments of the left is a subset of the right.
    Also works for nested types including ForwardRefs.

    Examples:
        Here are some code examples using `issubtype` from the `typing_utils` module:

        .. code-block:: python

            from typing_utils import issubtype

            # Examples of issubtype checks
            issubtype(typing.List, typing.Any)  # True
            issubtype(list, list)  # True
            issubtype(list, typing.List)  # True
            issubtype(list, typing.Sequence)  # True
            issubtype(typing.List[int], list)  # True
            issubtype(typing.List[typing.List], list)  # True
            issubtype(list, typing.List[int])  # False
            issubtype(list, typing.Union[typing.Tuple, typing.Set])  # False
            issubtype(typing.List[typing.List], typing.List[typing.Sequence])  # True

            # Example with custom JSON type
            JSON = typing.Union[
                int, float, bool, str, None, typing.Sequence["JSON"],
                typing.Mapping[str, "JSON"]
            ]
            issubtype(str, JSON, forward_refs={'JSON': JSON})  # True
            issubtype(typing.Dict[str, str], JSON, forward_refs={'JSON': JSON})  # True
            issubtype(typing.Dict[str, bytes], JSON, forward_refs={'JSON': JSON})  # False
    """
    return _is_normal_subtype(normalize(left), normalize(right), forward_refs)
