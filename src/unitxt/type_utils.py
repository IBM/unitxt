import ast
import collections.abc
import io
import itertools
import re
import typing

from .utils import safe_eval


def convert_union_type(type_string: str) -> str:
    """Converts Python 3.10 union type hints into form compatible with Python 3.9 version.

    Args:
        type_string (str): A string representation of a Python type hint. It can be any
            valid Python type, which does not contain strings (e.g. 'Literal').
            Examples include 'List[int|float]', 'str|float|bool' etc.

            Formally, the function depends on the input string adhering to the following rules.
            Assuming that the input is a valid type hint the function does not check that 'word' is
            'str', 'bool', 'List' etc. It just depends on the following general structure (spaces ignored):
            type -> word OR type( | type)* OR word[type( , type)*]
            word is a sequence of (0 or more) chars, each being any char but: [ ] , |
            This implies that if any of these 4 chars shows not as a meta char of the input
            type_string, but inside some constant string (of Literal, for example), the scheme
            will not work.

            Cases like Literal, that might contain occurrences of the four chars above not as meta chars
            in the type string, must be handled as special cases by this function, as shown for Literal,
            as an example. Because 'format_type_string' serves as preprocessing for 'parse_type_string',
            which has a list of allowed types, of which Literal is not a member, Literal and such are not
            relevant at all now; and the case is brought here just for an example for future use.


    Returns:
        str: A type string with converted union types, which is compatible with typing module.

    Examples:
        convert_union_type('List[int|float]') -> 'List[Union[int,float]]'
        convert_union_type('Optional[int|float|bool]') -> 'Optional[Union[int,float,bool]]'

    """

    def consume_literal(string: str) -> str:
        # identifies the prefix of string that matches a full Literal typing, with all its constants, including
        # constants that contain [ ] , etc. on which construct_union_part depends.
        # string starts with the [ that follows 'Literal'
        candidate_end = string.find("]")
        while candidate_end != -1:
            try:
                ast.literal_eval(string[: candidate_end + 1])
                break
            except Exception:
                candidate_end = string.find("]", candidate_end + 1)

        if candidate_end == -1:
            raise ValueError("invalid Literal in input type_string")
        return string[: candidate_end + 1]

    stack = [""]  # the start of a type
    input = type_string.strip()
    next_word = re.compile(r"([^\[\],|]*)([\[\],|]|$)")
    while len(input) > 0:
        word = next_word.match(input)
        input = input[len(word.group(0)) :].strip()
        stack[-1] += word.group(1)
        if word.group(2) in ["]", ",", ""]:  # "" for eol:$
            # top of stack is now complete to a whole type
            lwt = stack.pop()
            if (
                "|" in lwt
            ):  # the | -s are only at the top level of lwt, not inside any subtype
                lwt = "Union[" + lwt.replace("|", ",") + "]"
            lwt += word.group(2)
            if len(stack) > 0:
                stack[-1] += lwt
            else:
                stack = [lwt]
            if word.group(2) == ",":
                stack.append("")  # to start the expected next type

        elif word.group(2) in ["|"]:
            # top of stack is the last whole element(s) to be union-ed,
            # and more are expected
            stack[-1] += "|"

        else:  # "["
            if word.group(1) == "Literal":
                literal_ops = consume_literal("[" + input)
                stack[-1] += literal_ops
                input = input[len(literal_ops) - 1 :]
            else:
                stack[-1] += "["
                stack.append("")
                # start type (,type)*  inside the []

    assert len(stack) == 1
    if "|" in stack[0]:  # these belong to the top level only
        stack[0] = "Union[" + stack[0].replace("|", ",") + "]"
    return stack[0]


def format_type_string(type_string: str) -> str:
    """Formats a string representing a valid Python type hint so that it is compatible with Python 3.9 notation.

    Args:
        type_string (str): A string representation of a Python type hint. This can be any
                           valid type, which does not contain strings (e.g. 'Literal').
                           Examples include 'List[int]', 'Dict[str, Any]', 'Optional[List[str]]', etc.

    Returns:
        str: A formatted type string.

    Examples:
        format_type_string('list[int | float]') -> 'List[Union[int,float]]'
        format_type_string('dict[str, Optional[str]]') -> 'Dict[str,Optional[str]]'

    The function formats valid type string (either after or before Python 3.10) into a
    form compatible with 3.9. This is done by captilizing the first letter of a lower-cased
    type name and transferring the 'bitwise or operator' into 'Union' notation. The function
    also removes whitespaces and redundant module name in type names imported from 'typing'
    module, e.g. 'typing.Tuple' -> 'Tuple'.

    Currently, the capitalization is applied only to types which unitxt allows, i.e.
    'list', 'dict', 'tuple'. Moreover, the function expects the input to not contain types
    which contain strings, for example 'Literal'.
    """
    types_map = {
        "list": "List",
        "tuple": "Tuple",
        "dict": "Dict",
        "typing.": "",
        " ": "",
    }
    for old_type, new_type in types_map.items():
        type_string = type_string.replace(old_type, new_type)
    return convert_union_type(type_string)


def parse_type_string(type_string: str) -> typing.Any:
    """Parses a string representing a Python type hint and evaluates it to return the corresponding type object.

    This function uses a safe evaluation context
    to mitigate the risks of executing arbitrary code.

    Args:
        type_string (str): A string representation of a Python type hint. Examples include
                           'List[int]', 'Dict[str, Any]', 'Optional[List[str]]', etc.

    Returns:
        typing.Any: The Python type object corresponding to the given type string.

    Raises:
        ValueError: If the type string contains elements not allowed in the safe context
                    or tokens list.

    The function formats the string first if it represents a new Python type hint
    (i.e. valid since Python 3.10), which uses lowercased names for some types and
    'bitwise or operator' instead of 'Union', for example: 'list[int|float]' instead
    of 'List[Union[int,float]]' etc.

    The function uses a predefined safe context with common types from the `typing` module
    and basic Python data types. It also defines a list of safe tokens that are allowed
    in the type string.
    """
    safe_context = {
        "Any": typing.Any,
        "List": typing.List,
        "Dict": typing.Dict,
        "Tuple": typing.Tuple,
        "Union": typing.Union,
        "int": int,
        "str": str,
        "float": float,
        "bool": bool,
        "Optional": typing.Optional,
    }

    type_string = format_type_string(type_string)

    safe_tokens = ["[", "]", ",", " "]
    return safe_eval(type_string, safe_context, safe_tokens)


def infer_type(obj) -> typing.Any:
    return parse_type_string(infer_type_string(obj))


def infer_type_string(obj: typing.Any) -> str:
    """Encodes the type of a given object into a string.

    Args:
        obj:Any

    Returns:
        a string representation of the type of the object. e.g. 'str', 'List[int]', 'Dict[str, Any]'

    formal definition of the returned string:
    Type -> basic | List[Type] | Dict[Type, Type] | Union[Type (, Type)* | Tuple[Type (,Type)*]
    basic -> bool,str,int,float,Any
    no spaces at all.

    Examples:
        infer_type_string({"how_much": 7}) returns "Dict[str,int]"
        infer_type_string([1, 2]) returns "List[int]"
        infer_type_string([]) returns "List[Any]")    no contents to list to indicate any type
        infer_type_string([[], [7]]) returns "List[List[int]]"  type of parent list indicated by the type
                of the non-empty child list. The empty child list is indeed, by default, also of that type
                of the non-empty child.
        infer_type_string([[], 7, True]) returns "List[Union[List[Any],int]]"   because bool is also an int

    """

    def consume_arg(args_list: str) -> typing.Tuple[str, str]:
        first_word = re.search(r"^(List\[|Dict\[|Union\[|Tuple\[)", args_list)
        if not first_word:
            first_word = re.search(r"^(str|bool|int|float|Any)", args_list)
            assert first_word, "parsing error"
            return first_word.group(), args_list[first_word.span()[1] :]
        arg_to_ret = first_word.group()
        args_list = args_list[first_word.span()[1] :]
        arg, args_list = consume_arg(args_list)
        arg_to_ret += arg
        while args_list.startswith(","):
            arg, args_list = consume_arg(args_list[1:])
            arg_to_ret = arg_to_ret + "," + arg
        assert args_list.startswith("]"), "parsing error"
        return arg_to_ret + "]", args_list[1:]

    def find_args_in(args: str) -> typing.List[str]:
        to_ret = []
        while len(args) > 0:
            arg, args = consume_arg(args)
            to_ret.append(arg)
            if args.startswith(","):
                args = args[1:]
        return to_ret

    def is_covered_by(left: str, right: str) -> bool:
        if left == right:
            return True
        if left.startswith("Union["):
            return all(
                is_covered_by(left_el, right) for left_el in find_args_in(left[6:-1])
            )
        if right.startswith("Union["):
            return any(
                is_covered_by(left, right_el) for right_el in find_args_in(right[6:-1])
            )
        if left.startswith("List[") and right.startswith("List["):
            return is_covered_by(
                left[5:-1], right[5:-1]
            )  # un-wrap the leading List[  and the trailing ]
        if left.startswith("Dict[") and right.startswith("Dict["):
            return is_covered_by(
                left[5 : left.find(",")], right[5 : right.find(",")]
            ) and is_covered_by(
                left[1 + left.find(",") : -1], right[1 + right.find(",") : -1]
            )
        if left.startswith("Tuple[") and right.startswith("Tuple["):
            if left.count(",") != right.count(","):
                return False
            return all(
                is_covered_by(left_el, right_el)
                for (left_el, right_el) in zip(
                    left[6:-1].split(","), right[6:-1].split(",")
                )
            )
        if left == "bool" and right == "int":
            return True
        if left == "Any":
            return True

        return False

    def merge_into(left: str, right: typing.List[str]):
        # merge the set of types from left into the set of types from right, yielding a set that
        # covers both. None of the input sets contain Union as main element. Union may reside inside
        # List, or Dict, or Tuple.
        # This is needed when building a parent List, e.g. from its elements, and the
        # type of that list needs to be the union of the types of its elements.
        # if all elements have same type -- this is the type to write in List[type]
        # if not -- we write List[Union[type1, type2,...]].

        for right_el in right:
            if is_covered_by(right_el, left):
                right.remove(right_el)
                right.append(left)
                return
        if not any(is_covered_by(left, right_el) for right_el in right):
            right.append(left)

    def encode_a_list_of_type_names(list_of_type_names: typing.List[str]) -> str:
        # The type_names in the input are the set of names of all the elements of one list object,
        # or all the keys of one dict object, or all the val thereof, or all the type names of a specific position
        # in a tuple object The result should be a name of a type that covers them all.
        # So if, for example, the input contains both 'bool' and 'int', then 'int' suffices to cover both.
        # 'Any' can not show as a type_name of a basic (sub)object, but 'List[Any]' can show for an element of
        # a list object, an element that is an empty list. In such a case, if there are other elements in the input
        # that are more specific, e.g. 'List[str]' we should take the latter, and discard 'List[Any]' in order to get
        # a meaningful result: as narrow as possible but covers all.
        #
        to_ret = []
        for type_name in list_of_type_names:
            merge_into(type_name, to_ret)

        if len(to_ret) == 1:
            return to_ret[0]
        to_ret.sort()
        ans = "Union["
        for typ in to_ret[:-1]:
            ans += typ + ","
        return ans + to_ret[-1] + "]"

    basic_types = [bool, int, str, float]
    names_of_basic_types = ["bool", "int", "str", "float"]
    # bool should show before int, because bool is subtype of int

    for basic_type, name_of_basic_type in zip(basic_types, names_of_basic_types):
        if isinstance(obj, basic_type):
            return name_of_basic_type
    if isinstance(obj, list):
        included_types = set()
        for list_el in obj:
            included_types.add(infer_type_string(list_el))
        included_types = list(included_types)
        if len(included_types) == 0:
            return "List[Any]"
        return "List[" + encode_a_list_of_type_names(included_types) + "]"
    if isinstance(obj, dict):
        if len(obj) == 0:
            return "Dict[Any,Any]"
        included_key_types = set()
        included_val_types = set()
        for k, v in obj.items():
            included_key_types.add(infer_type_string(k))
            included_val_types.add(infer_type_string(v))
        included_key_types = list(included_key_types)
        included_val_types = list(included_val_types)
        return (
            "Dict["
            + encode_a_list_of_type_names(included_key_types)
            + ","
            + encode_a_list_of_type_names(included_val_types)
            + "]"
        )
    if isinstance(obj, tuple):
        if len(obj) == 0:
            return "Tuple[Any]"
        to_ret = "Tuple["
        for sub_tup in obj[:-1]:
            to_ret += infer_type_string(sub_tup) + ","
        return to_ret + infer_type_string(obj[-1]) + "]"

    return "Any"


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


def to_float_or_default(v, failure_default=0):
    try:
        return float(v)
    except Exception as e:
        if failure_default is None:
            raise e
        return failure_default


def verify_required_schema(
    required_schema_dict: typing.Dict[str, str],
    input_dict: typing.Dict[str, typing.Any],
) -> None:
    """Verifies if passed input_dict has all required fields, and they are of proper types according to required_schema_dict.

    Parameters:
        required_schema_dict (Dict[str, str]):
            Schema where a key is name of a field and a value is a string
            representing a type of its value.
        input_dict (Dict[str, Any]):
            Dict with input fields and their respective values.
    """
    for field_name, data_type_string in required_schema_dict.items():
        try:
            value = input_dict[field_name]
        except KeyError as e:
            raise KeyError(
                f"Unexpected field name: '{field_name}'. "
                f"The available names: {list(input_dict.keys())}."
            ) from e

        data_type = parse_type_string(data_type_string)

        if not isoftype(value, data_type):
            raise ValueError(
                f"Passed value '{value}' of field '{field_name}' is not "
                f"of required type: ({data_type_string})."
            )
