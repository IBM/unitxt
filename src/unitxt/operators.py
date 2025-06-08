"""This section describes unitxt operators.

Operators: Building Blocks of Unitxt Processing Pipelines
==============================================================

Within the Unitxt framework, operators serve as the foundational elements used to assemble processing pipelines.
Each operator is designed to perform specific manipulations on dictionary structures within a stream.
These operators are callable entities that receive a MultiStream as input.
The output is a MultiStream, augmented with the operator's manipulations, which are then systematically applied to each instance in the stream when pulled.

Creating Custom Operators
-------------------------------
To enhance the functionality of Unitxt, users are encouraged to develop custom operators.
This can be achieved by inheriting from any of the existing operators listed below or from one of the fundamental :class:`base operators<unitxt.operator>`.
The primary task in any operator development is to implement the `process` function, which defines the unique manipulations the operator will perform.

General or Specialized Operators
--------------------------------
Some operators are specialized in specific data or specific operations such as:

- :class:`loaders<unitxt.loaders>` for accessing data from various sources.
- :class:`splitters<unitxt.splitters>` for fixing data splits.
- :class:`stream_operators<unitxt.stream_operators>` for changing joining and mixing streams.
- :class:`struct_data_operators<unitxt.struct_data_operators>` for structured data operators.
- :class:`collections_operators<unitxt.collections_operators>` for handling collections such as lists and dictionaries.
- :class:`dialog_operators<unitxt.dialog_operators>` for handling dialogs.
- :class:`string_operators<unitxt.string_operators>` for handling strings.
- :class:`span_labeling_operators<unitxt.span_lableing_operators>` for handling strings.
- :class:`fusion<unitxt.fusion>` for fusing and mixing datasets.

Other specialized operators are used by unitxt internally:

- :class:`templates<unitxt.templates>` for verbalizing data examples.
- :class:`formats<unitxt.formats>` for preparing data for models.

The rest of this section is dedicated to general operators.

General Operators List:
------------------------
"""

import operator
import uuid
import warnings
import zipfile
from abc import abstractmethod
from collections import Counter, defaultdict
from dataclasses import field
from itertools import zip_longest
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import requests

from .artifact import Artifact, fetch_artifact
from .dataclass import NonPositionalField, OptionalField
from .deprecation_utils import deprecation
from .dict_utils import dict_delete, dict_get, dict_set, is_subpath
from .error_utils import UnitxtError
from .generator_utils import ReusableGenerator
from .operator import (
    InstanceOperator,
    MultiStream,
    MultiStreamOperator,
    PagedStreamOperator,
    SequentialOperator,
    SideEffectOperator,
    SourceOperator,
    StreamingOperator,
    StreamInitializerOperator,
    StreamOperator,
)
from .random_utils import new_random_generator
from .settings_utils import get_settings
from .stream import DynamicStream, Stream
from .text_utils import to_pretty_string
from .type_utils import isoftype
from .utils import (
    LRUCache,
    deep_copy,
    flatten_dict,
    recursive_copy,
    recursive_shallow_copy,
    shallow_copy,
)

settings = get_settings()


class FromIterables(StreamInitializerOperator):
    """Creates a MultiStream from a dict of named iterables.

    Example:
        operator = FromIterables()
        ms = operator.process(iterables)

    """

    def process(self, iterables: Dict[str, Iterable]) -> MultiStream:
        return MultiStream.from_iterables(iterables)


class IterableSource(SourceOperator):
    """Creates a MultiStream from a dict of named iterables.

    It is a callable.

    Args:
        iterables (Dict[str, Iterable]): A dictionary mapping stream names to iterables.

    Example:
        operator =  IterableSource(input_dict)
        ms = operator()

    """

    iterables: Dict[str, Iterable]

    def process(self) -> MultiStream:
        return MultiStream.from_iterables(self.iterables)


class MapInstanceValues(InstanceOperator):
    """A class used to map instance values into other values.

    This class is a type of ``InstanceOperator``,
    it maps values of instances in a stream using predefined mappers.

    Args:
        mappers (Dict[str, Dict[str, Any]]):
            The mappers to use for mapping instance values.
            Keys are the names of the fields to undergo mapping, and values are dictionaries
            that define the mapping from old values to new values.
            Note that mapped values are defined by their string representation, so mapped values
            are converted to strings before being looked up in the mappers.
        strict (bool):
            If True, the mapping is applied strictly. That means if a value
            does not exist in the mapper, it will raise a KeyError. If False, values
            that are not present in the mapper are kept as they are.
        process_every_value (bool):
            If True, all fields to be mapped should be lists, and the mapping
            is to be applied to their individual elements.
            If False, mapping is only applied to a field containing a single value.

    Examples:
        ``MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}})``
        replaces ``"1"`` with ``"hi"`` and ``"2"`` with ``"bye"`` in field ``"a"`` in all instances of all streams:
        instance ``{"a": 1, "b": 2}`` becomes ``{"a": "hi", "b": 2}``. Note that the value of ``"b"`` remained intact,
        since field-name ``"b"`` does not participate in the mappers, and that ``1`` was casted to ``"1"`` before looked
        up in the mapper of ``"a"``.

        ``MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}, process_every_value=True)``:
        Assuming field ``"a"`` is a list of values, potentially including ``"1"``-s and ``"2"``-s, this replaces
        each such ``"1"`` with ``"hi"`` and ``"2"`` -- with ``"bye"`` in all instances of all streams:
        instance ``{"a": ["1", "2"], "b": 2}`` becomes ``{"a": ["hi", "bye"], "b": 2}``.

        ``MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}, strict=True)``:
        To ensure that all values of field ``"a"`` are mapped in every instance, use ``strict=True``.
        Input instance ``{"a":"3", "b": 2}`` will raise an exception per the above call,
        because ``"3"`` is not a key in the mapper of ``"a"``.

        ``MapInstanceValues(mappers={"a": {str([1,2,3,4]): "All", str([]): "None"}}, strict=True)``
        replaces a list ``[1,2,3,4]`` with the string ``"All"`` and an empty list by string ``"None"``.

    """

    mappers: Dict[str, Dict[str, str]]
    strict: bool = True
    process_every_value: bool = False

    def verify(self):
        # make sure the mappers are valid
        for key, mapper in self.mappers.items():
            assert isinstance(
                mapper, dict
            ), f"Mapper for given field {key} should be a dict, got {type(mapper)}"
            for k in mapper.keys():
                assert isinstance(
                    k, str
                ), f'Key "{k}" in mapper for field "{key}" should be a string, got {type(k)}'

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for key, mapper in self.mappers.items():
            value = dict_get(instance, key)
            if value is not None:
                if (self.process_every_value is True) and (not isinstance(value, list)):
                    raise ValueError(
                        f"'process_every_field' == True is allowed only for fields whose values are lists, but value of field '{key}' is '{value}'"
                    )
                if isinstance(value, list) and self.process_every_value:
                    for i, val in enumerate(value):
                        value[i] = self.get_mapped_value(instance, key, mapper, val)
                else:
                    value = self.get_mapped_value(instance, key, mapper, value)
                dict_set(
                    instance,
                    key,
                    value,
                )

        return instance

    def get_mapped_value(self, instance, key, mapper, val):
        val_as_str = str(val)  # make sure the value is a string
        if val_as_str in mapper:
            return recursive_copy(mapper[val_as_str])
        if self.strict:
            raise KeyError(
                f"value '{val_as_str}', the string representation of the value in field '{key}', is not found in mapper '{mapper}'"
            )
        return val


class FlattenInstances(InstanceOperator):
    """Flattens each instance in a stream, making nested dictionary entries into top-level entries.

    Args:
        parent_key (str): A prefix to use for the flattened keys. Defaults to an empty string.
        sep (str): The separator to use when concatenating nested keys. Defaults to "_".
    """

    parent_key: str = ""
    sep: str = "_"

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return flatten_dict(instance, parent_key=self.parent_key, sep=self.sep)


class Set(InstanceOperator):
    """Sets specified fields in each instance, in a given stream or all streams (default), with specified values. If fields exist, updates them, if do not exist -- adds them.

    Args:
        fields (Dict[str, object]): The fields to add to each instance. Use '/' to access inner fields

        use_deepcopy (bool) : Deep copy the input value to avoid later modifications

    Examples:
        # Set a value of a list consisting of "positive" and "negative" do field "classes" to each and every instance of all streams
        ``Set(fields={"classes": ["positive","negatives"]})``

        # In each and every instance of all streams, field "span" is to become a dictionary containing a field "start", in which the value 0 is to be set
        ``Set(fields={"span/start": 0}``

        # In all instances of stream "train" only, Set field "classes" to have the value of a list consisting of "positive" and "negative"
        ``Set(fields={"classes": ["positive","negatives"], apply_to_stream=["train"]})``

        # Set field "classes" to have the value of a given list, preventing modification of original list from changing the instance.
        ``Set(fields={"classes": alist}), use_deepcopy=True)``  if now alist is modified, still the instances remain intact.
    """

    fields: Dict[str, object]
    use_query: Optional[bool] = None
    use_deepcopy: bool = False

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for key, value in self.fields.items():
            if self.use_deepcopy:
                value = deep_copy(value)
            dict_set(instance, key, value)
        return instance


def recursive_key_value_replace(data, target_key, value_map, value_remove=None):
    """Recursively traverses a data structure (dicts and lists), replaces values of target_key using value_map, and removes values listed in value_remove.

    Args:
        data: The data structure (dict or list) to traverse.
        target_key: The specific key whose value needs to be checked and replaced or removed.
        value_map: A dictionary mapping old values to new values.
        value_remove: A list of values to completely remove if found as values of target_key.

    Returns:
        The modified data structure. Modification is done in-place.
    """
    if value_remove is None:
        value_remove = []

    if isinstance(data, dict):
        keys_to_delete = []
        for key, value in data.items():
            if key == target_key:
                if isinstance(value, list):
                    data[key] = [
                        value_map.get(item, item)
                        for item in value
                        if not isinstance(item, dict) and item not in value_remove
                    ]
                elif isinstance(value, dict):
                    pass  # Skip or handle dict values if needed
                elif value in value_remove:
                    keys_to_delete.append(key)
                elif value in value_map:
                    data[key] = value_map[value]
            else:
                recursive_key_value_replace(value, target_key, value_map, value_remove)
        for key in keys_to_delete:
            del data[key]
    elif isinstance(data, list):
        for item in data:
            recursive_key_value_replace(item, target_key, value_map, value_remove)
    return data


class RecursiveReplace(InstanceOperator):
    # Assisted by watsonx Code Assistant
    """An operator to recursively replace values in dictionary fields of instances based on a key and a mapping of values.

    Attributes:
        key (str): The key in the dictionary to start the replacement process.
        map_values (dict): A dictionary containing the key-value pairs to replace the original values.
        remove_values (Optional[list]): An optional list of values to remove from the dictionary. Defaults to None.

    Example:
    RecursiveReplace(key="a", map_values={"1": "hi", "2": "bye" }, remove_values=["3"])
        replaces the value of key "a" in all instances of all streams:
        instance ``{"field" : [{"a": "1", "b" : "2"}, {"a" : "3", "b:" "4"}}` becomes ``{"field" : [{"a": "hi", "b" : "2"}, {"b": "4"}}``

        Notice how the value of field ``"a"`` in the first instance is replaced with ``"hi"`` and the value of field ``"a"`` in the second instance is removed.
    """

    key: str
    map_values: dict
    remove_values: Optional[list] = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return recursive_key_value_replace(
            instance, self.key, self.map_values, self.remove_values
        )


@deprecation(version="2.0.0", alternative=Set)
class AddFields(Set):
    pass


class RemoveFields(InstanceOperator):
    """Remove specified fields from each instance in a stream.

    Args:
        fields (List[str]): The fields to remove from each instance.
    """

    fields: List[str]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field_name in self.fields:
            del instance[field_name]
        return instance


class SelectFields(InstanceOperator):
    """Keep only specified fields from each instance in a stream.

    Args:
        fields (List[str]): The fields to keep from each instance.
    """

    fields: List[str]

    def prepare(self):
        super().prepare()
        self.fields.extend(["data_classification_policy", "recipe_metadata"])

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        new_instance = {}
        for selected_field in self.fields:
            new_instance[selected_field] = instance[selected_field]
        return new_instance


class DefaultPlaceHolder:
    pass


default_place_holder = DefaultPlaceHolder()


class InstanceFieldOperator(InstanceOperator):
    """A general stream instance operator that processes the values of a field (or multiple ones).

    Args:
        field (Optional[str]):
            The field to process, if only a single one is passed. Defaults to None
        to_field (Optional[str]):
            Field name to save result into, if only one field is processed, if None is passed the
            operation would happen in-place and its result would replace the value of ``field``. Defaults to None
        field_to_field (Optional[Union[List[List[str]], Dict[str, str]]]):
            Mapping from names of fields to process,
            to names of fields to save the results into. Inner List, if used, should be of length 2.
            A field is processed by feeding its value into method ``process_value`` and storing the result in ``to_field`` that
            is mapped to the field. When the type of argument ``field_to_field`` is List, the order by which the fields are processed is their order
            in the (outer) List. But when the type of argument ``field_to_field`` is Dict, there is no uniquely determined
            order. The end result might depend on that order if either (1) two different fields are mapped to the same
            to_field, or (2) a field shows both as a key and as a value in different mappings.
            The operator throws an AssertionError in either of these cases. ``field_to_field``
            defaults to None.
        process_every_value (bool):
            Processes the values in a list instead of the list as a value, similar to python's ``*var``. Defaults to False

    Note: if ``field`` and ``to_field`` (or both members of a pair in ``field_to_field`` ) are equal (or share a common
    prefix if ``field`` and ``to_field`` contain a / ), then the result of the operation is saved within ``field`` .

    """

    field: Optional[str] = None
    to_field: Optional[str] = None
    field_to_field: Optional[Union[List[List[str]], Dict[str, str]]] = None
    use_query: Optional[bool] = None
    process_every_value: bool = False
    get_default: Any = None
    not_exist_ok: bool = False
    not_exist_do_nothing: bool = False

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def verify_field_definition(self):
        if hasattr(self, "_field_to_field") and self._field_to_field is not None:
            return
        assert (
            (self.field is None) != (self.field_to_field is None)
        ), "Must uniquely define the field to work on, through exactly one of either 'field' or 'field_to_field'"
        assert (
            self.to_field is None or self.field_to_field is None
        ), f"Can not apply operator to create both {self.to_field} and the to fields in the mapping {self.field_to_field}"

        if self.field_to_field is None:
            self._field_to_field = [
                (self.field, self.to_field if self.to_field is not None else self.field)
            ]
        else:
            self._field_to_field = (
                list(self.field_to_field.items())
                if isinstance(self.field_to_field, dict)
                else self.field_to_field
            )
        assert (
            self.field is not None or self.field_to_field is not None
        ), "Must supply a field to work on"
        assert (
            self.to_field is None or self.field_to_field is None
        ), f"Can not apply operator to create both on {self.to_field} and on the mapping from fields to fields {self.field_to_field}"
        assert (
            self.field is None or self.field_to_field is None
        ), f"Can not apply operator both on {self.field} and on the from fields in the mapping {self.field_to_field}"
        assert (
            self._field_to_field is not None
        ), f"the from and to fields must be defined or implied from the other inputs got: {self._field_to_field}"
        assert (
            len(self._field_to_field) > 0
        ), f"'input argument '{self.__class__.__name__}.field_to_field' should convey at least one field to process. Got {self.field_to_field}"
        # self._field_to_field is built explicitly by pairs, or copied from argument 'field_to_field'
        if self.field_to_field is None:
            return
        # for backward compatibility also allow list of tuples of two strings
        if isoftype(self.field_to_field, List[List[str]]) or isoftype(
            self.field_to_field, List[Tuple[str, str]]
        ):
            for pair in self._field_to_field:
                assert (
                    len(pair) == 2
                ), f"when 'field_to_field' is defined as a list of lists, the inner lists should all be of length 2. {self.field_to_field}"
            # order of field processing is uniquely determined by the input field_to_field when a list
            return
        if isoftype(self.field_to_field, Dict[str, str]):
            if len(self.field_to_field) < 2:
                return
            for ff, tt in self.field_to_field.items():
                for f, t in self.field_to_field.items():
                    if f == ff:
                        continue
                    assert (
                        t != ff
                    ), f"In input argument 'field_to_field': {self.field_to_field}, field {f} is mapped to field {t}, while the latter is mapped to {tt}. Whether {f} or {t} is processed first might impact end result."
                    assert (
                        tt != t
                    ), f"In input argument 'field_to_field': {self.field_to_field}, two different fields: {ff} and {f} are mapped to field {tt}. Whether {ff} or {f} is processed last might impact end result."
            return
        raise ValueError(
            "Input argument 'field_to_field': {self.field_to_field} is neither of type List{List[str]] nor of type Dict[str, str]."
        )

    @abstractmethod
    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        pass

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        self.verify_field_definition()
        for from_field, to_field in self._field_to_field:
            try:
                old_value = dict_get(
                    instance,
                    from_field,
                    default=default_place_holder,
                    not_exist_ok=self.not_exist_ok or self.not_exist_do_nothing,
                )
                if old_value is default_place_holder:
                    if self.not_exist_do_nothing:
                        continue
                    old_value = self.get_default
            except Exception as e:
                raise ValueError(
                    f"Failed to get '{from_field}' from instance due to the exception above."
                ) from e
            try:
                if self.process_every_value:
                    new_value = [
                        self.process_instance_value(value, instance)
                        for value in old_value
                    ]
                else:
                    new_value = self.process_instance_value(old_value, instance)
            except Exception as e:
                raise ValueError(
                    f"Failed to process field '{from_field}' from instance due to the exception above."
                ) from e
            dict_set(
                instance,
                to_field,
                new_value,
                not_exist_ok=True,
            )
        return instance


class FieldOperator(InstanceFieldOperator):
    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        return self.process_value(value)

    @abstractmethod
    def process_value(self, value: Any) -> Any:
        pass


class MapValues(FieldOperator):
    mapping: Dict[str, str]

    def process_value(self, value: Any) -> Any:
        return self.mapping[str(value)]


class Rename(FieldOperator):
    """Renames fields.

    Move value from one field to another, potentially, if field name contains a /, from one branch into another.
    Remove the from field, potentially part of it in case of / in from_field.

    Examples:
        Rename(field_to_field={"b": "c"})
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "c": 2}, {"a": 2, "c": 3}]

        Rename(field_to_field={"b": "c/d"})
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "c": {"d": 2}}, {"a": 2, "c": {"d": 3}}]

        Rename(field_to_field={"b": "b/d"})
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "b": {"d": 2}}, {"a": 2, "b": {"d": 3}}]

        Rename(field_to_field={"b/c/e": "b/d"})
        will change inputs [{"a": 1, "b": {"c": {"e": 2, "f": 20}}}] to [{"a": 1, "b": {"c": {"f": 20}, "d": 2}}]

    """

    def process_value(self, value: Any) -> Any:
        return value

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        res = super().process(instance=instance, stream_name=stream_name)
        for from_field, to_field in self._field_to_field:
            if (not is_subpath(from_field, to_field)) and (
                not is_subpath(to_field, from_field)
            ):
                dict_delete(res, from_field, remove_empty_ancestors=True)

        return res


@deprecation(version="2.0.0", alternative=Rename)
class RenameFields(Rename):
    pass


class AddConstant(FieldOperator):
    """Adds a constant, being argument 'add', to the processed value.

    Args:
        add: the constant to add.
    """

    add: Any

    def process_value(self, value: Any) -> Any:
        return self.add + value


class ShuffleFieldValues(FieldOperator):
    # Assisted by watsonx Code Assistant
    """An operator that shuffles the values of a list field.

    the seed for shuffling in the is determined by the elements of the input field,
    ensuring that the shuffling operation produces different results for different input lists,
    but also that it is deterministic and reproducible.

    Attributes:
        None

    Methods:
        process_value(value: Any) -> Any:
            Shuffles the elements of the input list and returns the shuffled list.

            Parameters:
                value (Any): The input list to be shuffled.

    Returns:
                Any: The shuffled list.
    """

    def process_value(self, value: Any) -> Any:
        res = list(value)
        random_generator = new_random_generator(sub_seed=res)
        random_generator.shuffle(res)
        return res


class JoinStr(FieldOperator):
    """Joins a list of strings (contents of a field), similar to str.join().

    Args:
        separator (str): text to put between values
    """

    separator: str = ","

    def process_value(self, value: Any) -> Any:
        return self.separator.join(str(x) for x in value)


class Apply(InstanceOperator):
    """A class used to apply a python function and store the result in a field.

    Args:
        function (str): name of function.
        to_field (str): the field to store the result

    any additional arguments are field names whose values will be passed directly to the function specified

    Examples:
    Store in field  "b" the uppercase string of the value in field "a":
    ``Apply("a", function=str.upper, to_field="b")``

    Dump the json representation of field "t" and store back in the same field:
    ``Apply("t", function=json.dumps, to_field="t")``

    Set the time in a field 'b':
    ``Apply(function=time.time, to_field="b")``

    """

    __allow_unexpected_arguments__ = True
    function: Callable = NonPositionalField(required=True)
    to_field: str = NonPositionalField(required=True)

    def function_to_str(self, function: Callable) -> str:
        parts = []

        if hasattr(function, "__module__"):
            parts.append(function.__module__)
        if hasattr(function, "__qualname__"):
            parts.append(function.__qualname__)
        else:
            parts.append(function.__name__)

        return ".".join(parts)

    def str_to_function(self, function_str: str) -> Callable:
        parts = function_str.split(".", 1)
        if len(parts) == 1:
            return __builtins__[parts[0]]

        module_name, function_name = parts
        if module_name in __builtins__:
            obj = __builtins__[module_name]
        elif module_name in globals():
            obj = globals()[module_name]
        else:
            obj = __import__(module_name)
        for part in function_name.split("."):
            obj = getattr(obj, part)
        return obj

    def prepare(self):
        super().prepare()
        if isinstance(self.function, str):
            self.function = self.str_to_function(self.function)
        self._init_dict["function"] = self.function_to_str(self.function)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        argv = [instance[arg] for arg in self._argv]
        kwargs = {key: instance[val] for key, val in self._kwargs}

        result = self.function(*argv, **kwargs)

        instance[self.to_field] = result
        return instance


class ListFieldValues(InstanceOperator):
    """Concatenates values of multiple fields into a list, and assigns it to a new field."""

    fields: List[str]
    to_field: str
    use_query: Optional[bool] = None

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        values = []
        for field_name in self.fields:
            values.append(dict_get(instance, field_name))

        dict_set(instance, self.to_field, values)

        return instance


class ZipFieldValues(InstanceOperator):
    """Zips values of multiple fields in a given instance, similar to ``list(zip(*fields))``.

    The value in each of the specified 'fields' is assumed to be a list. The lists from all 'fields'
    are zipped, and stored into 'to_field'.

    | If 'longest'=False, the length of the zipped result is determined by the shortest input value.
    | If 'longest'=True, the length of the zipped result is determined by the longest input, padding shorter inputs with None-s.

    """

    fields: List[str]
    to_field: str
    longest: bool = False
    use_query: Optional[bool] = None

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        values = []
        for field_name in self.fields:
            values.append(dict_get(instance, field_name))
        if self.longest:
            zipped = zip_longest(*values)
        else:
            zipped = zip(*values)
        dict_set(instance, self.to_field, list(zipped))
        return instance


class InterleaveListsToDialogOperator(InstanceOperator):
    """Interleaves two lists, one of user dialog turns and one of assistant dialog turns, into a single list of tuples, alternating between "user" and "assistant".

    The list of tuples if of format (role, turn_content), where the role label is specified by
    the 'user_role_label' and 'assistant_role_label' fields (default to "user" and "assistant").

    The user turns and assistant turns field are specified in the arguments.
    The value of each of the 'fields' is assumed to be a list.

    """

    user_turns_field: str
    assistant_turns_field: str
    user_role_label: str = "user"
    assistant_role_label: str = "assistant"
    to_field: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        user_turns = instance[self.user_turns_field]
        assistant_turns = instance[self.assistant_turns_field]

        assert (
            len(user_turns) == len(assistant_turns)
            or (len(user_turns) - len(assistant_turns) == 1)
        ), "user_turns must have either the same length as assistant_turns or one more turn."

        interleaved_dialog = []
        i, j = 0, 0  # Indices for the user and assistant lists
        # While either list has elements left, continue interleaving
        while i < len(user_turns) or j < len(assistant_turns):
            if i < len(user_turns):
                interleaved_dialog.append((self.user_role_label, user_turns[i]))
                i += 1
            if j < len(assistant_turns):
                interleaved_dialog.append(
                    (self.assistant_role_label, assistant_turns[j])
                )
                j += 1

        instance[self.to_field] = interleaved_dialog
        return instance


class IndexOf(InstanceOperator):
    """For a given instance, finds the offset of value of field 'index_of', within the value of field 'search_in'."""

    search_in: str
    index_of: str
    to_field: str
    use_query: Optional[bool] = None

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        lst = dict_get(instance, self.search_in)
        item = dict_get(instance, self.index_of)
        instance[self.to_field] = lst.index(item)
        return instance


class TakeByField(InstanceOperator):
    """From field 'field' of a given instance, select the member indexed by field 'index', and store to field 'to_field'."""

    field: str
    index: str
    to_field: str = None
    use_query: Optional[bool] = None

    def verify(self):
        super().verify()
        if self.use_query is not None:
            depr_message = "Field 'use_query' is deprecated. From now on, default behavior is compatible to use_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def prepare(self):
        if self.to_field is None:
            self.to_field = self.field

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        value = dict_get(instance, self.field)
        index_value = dict_get(instance, self.index)
        instance[self.to_field] = value[index_value]
        return instance


class Perturb(FieldOperator):
    """Slightly perturbs the contents of ``field``. Could be Handy for imitating prediction from given target.

    When task was classification, argument ``select_from`` can be used to list the other potential classes, as a
    relevant perturbation

    Args:
        percentage_to_perturb (int):
            the percentage of the instances for which to apply this perturbation. Defaults to 1 (1 percent)
        select_from: List[Any]:
            a list of values to select from, as a perturbation of the field's value. Defaults to [].
    """

    select_from: List[Any] = []
    percentage_to_perturb: int = 1  # 1 percent

    def verify(self):
        assert (
            0 <= self.percentage_to_perturb and self.percentage_to_perturb <= 100
        ), f"'percentage_to_perturb' should be in the range 0..100. Received {self.percentage_to_perturb}"

    def prepare(self):
        super().prepare()
        self.random_generator = new_random_generator(sub_seed="CopyWithPerturbation")

    def process_value(self, value: Any) -> Any:
        perturb = self.random_generator.randint(1, 100) <= self.percentage_to_perturb
        if not perturb:
            return value

        if value in self.select_from:
            # 80% of cases, return a decent class, otherwise, perturb the value itself as follows
            if self.random_generator.random() < 0.8:
                return self.random_generator.choice(self.select_from)

        if isinstance(value, float):
            return value * (0.5 + self.random_generator.random())

        if isinstance(value, int):
            perturb = 1 if self.random_generator.random() < 0.5 else -1
            return value + perturb

        if isinstance(value, str):
            if len(value) < 2:
                # give up perturbation
                return value
            # throw one char out
            prefix_len = self.random_generator.randint(1, len(value) - 1)
            return value[:prefix_len] + value[prefix_len + 1 :]

        # and in any other case:
        return value


class Copy(FieldOperator):
    """Copies values from specified fields to specified fields.

    Args (of parent class):
        field_to_field (Union[List[List], Dict[str, str]]): A list of lists, where each sublist contains the source field and the destination field, or a dictionary mapping source fields to destination fields.

    Examples:
        An input instance {"a": 2, "b": 3}, when processed by
        ``Copy(field_to_field={"a": "b"})``
        would yield {"a": 2, "b": 2}, and when processed by
        ``Copy(field_to_field={"a": "c"})`` would yield
        {"a": 2, "b": 3, "c": 2}

        with field names containing / , we can also copy inside the field:
        ``Copy(field="a/0",to_field="a")``
        would process instance {"a": [1, 3]} into {"a": 1}


    """

    def process_value(self, value: Any) -> Any:
        return value


class RecursiveCopy(FieldOperator):
    def process_value(self, value: Any) -> Any:
        return recursive_copy(value)


@deprecation(version="2.0.0", alternative=Copy)
class CopyFields(Copy):
    pass


class GetItemByIndex(FieldOperator):
    """Get the element from the fixed list by the index in the given field and store in another field.

    Example:
        GetItemByIndex(items_list=["dog",cat"],field="animal_index",to_field="animal")

    on instance {"animal_index" : 1}  will change the instance to {"animal_index" : 1, "animal" : "cat"}

    """

    items_list: List[Any]

    def process_value(self, value: Any) -> Any:
        return self.items_list[value]


class AddID(InstanceOperator):
    """Stores a unique id value in the designated 'id_field_name' field of the given instance."""

    id_field_name: str = "id"

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance[self.id_field_name] = str(uuid.uuid4()).replace("-", "")
        return instance


class Cast(FieldOperator):
    """Casts specified fields to specified types.

    Args:
        default (object): A dictionary mapping field names to default values for cases of casting failure.
        process_every_value (bool): If true, all fields involved must contain lists, and each value in the list is then casted. Defaults to False.
    """

    to: str
    failure_default: Optional[Any] = "__UNDEFINED__"

    def prepare(self):
        self.types = {
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "tuple": tuple,
        }

    def process_value(self, value):
        try:
            return self.types[self.to](value)
        except ValueError as e:
            if self.failure_default == "__UNDEFINED__":
                raise ValueError(
                    f'Failed to cast value {value} to type "{self.to}", and no default value is provided.'
                ) from e
            return self.failure_default


class CastFields(InstanceOperator):
    """Casts specified fields to specified types.

    Args:
        fields (Dict[str, str]):
            A dictionary mapping field names to the names of the types to cast the fields to.
            e.g: "int", "str", "float", "bool". Basic names of types
        defaults (Dict[str, object]):
            A dictionary mapping field names to default values for cases of casting failure.
        process_every_value (bool):
            If true, all fields involved must contain lists, and each value in the list is then casted. Defaults to False.

    Example:
        .. code-block:: python

                CastFields(
                    fields={"a/d": "float", "b": "int"},
                    failure_defaults={"a/d": 0.0, "b": 0},
                    process_every_value=True,
                )

    would process the input instance: ``{"a": {"d": ["half", "0.6", 1, 12]}, "b": ["2"]}``
    into ``{"a": {"d": [0.0, 0.6, 1.0, 12.0]}, "b": [2]}``.

    """

    fields: Dict[str, str] = field(default_factory=dict)
    failure_defaults: Dict[str, object] = field(default_factory=dict)
    use_nested_query: bool = None  # deprecated field
    process_every_value: bool = False

    def prepare(self):
        self.types = {"int": int, "float": float, "str": str, "bool": bool}

    def verify(self):
        super().verify()
        if self.use_nested_query is not None:
            depr_message = "Field 'use_nested_query' is deprecated. From now on, default behavior is compatible to use_nested_query=True. Please remove this field from your code."
            warnings.warn(depr_message, DeprecationWarning, stacklevel=2)

    def _cast_single(self, value, type, field):
        try:
            return self.types[type](value)
        except Exception as e:
            if field not in self.failure_defaults:
                raise ValueError(
                    f'Failed to cast field "{field}" with value {value} to type "{type}", and no default value is provided.'
                ) from e
            return self.failure_defaults[field]

    def _cast_multiple(self, values, type, field):
        return [self._cast_single(value, type, field) for value in values]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field_name, type in self.fields.items():
            value = dict_get(instance, field_name)
            if self.process_every_value:
                assert isinstance(
                    value, list
                ), f"'process_every_field' == True is allowed only for fields whose values are lists, but value of field '{field_name}' is '{value}'"
                casted_value = self._cast_multiple(value, type, field_name)
            else:
                casted_value = self._cast_single(value, type, field_name)

            dict_set(instance, field_name, casted_value)
        return instance


class DivideAllFieldsBy(InstanceOperator):
    """Recursively reach down to all fields that are float, and divide each by 'divisor'.

    The given instance is viewed as a tree whose internal nodes are dictionaries and lists, and
    the leaves are either 'float' and then divided, or other basic type, in which case, a ValueError is raised
    if input flag 'strict' is True, or -- left alone, if 'strict' is False.

    Args:
        divisor (float) the value to divide by
        strict (bool) whether to raise an error upon visiting a leaf that is not float. Defaults to False.

    Example:
        when instance {"a": 10.0, "b": [2.0, 4.0, 7.0], "c": 5} is processed by operator:
        operator = DivideAllFieldsBy(divisor=2.0)
        the output is: {"a": 5.0, "b": [1.0, 2.0, 3.5], "c": 5}
        If the operator were defined with strict=True, through:
        operator = DivideAllFieldsBy(divisor=2.0, strict=True),
        the processing of the above instance would raise a ValueError, for the integer at "c".
    """

    divisor: float = 1.0
    strict: bool = False

    def _recursive_divide(self, instance, divisor):
        if isinstance(instance, dict):
            for key, value in instance.items():
                instance[key] = self._recursive_divide(value, divisor)
        elif isinstance(instance, list):
            for i, value in enumerate(instance):
                instance[i] = self._recursive_divide(value, divisor)
        elif isinstance(instance, float):
            instance /= divisor
        elif self.strict:
            raise ValueError(f"Cannot divide instance of type {type(instance)}")
        return instance

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return self._recursive_divide(instance, self.divisor)


class ArtifactFetcherMixin:
    """Provides a way to fetch and cache artifacts in the system.

    Args:
        cache (Dict[str, Artifact]): A cache for storing fetched artifacts.
    """

    _artifacts_cache = LRUCache(max_size=1000)

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if str(artifact_identifier) not in cls._artifacts_cache:
            artifact, catalog = fetch_artifact(artifact_identifier)
            cls._artifacts_cache[str(artifact_identifier)] = artifact
        return shallow_copy(cls._artifacts_cache[str(artifact_identifier)])


class ApplyOperatorsField(InstanceOperator):
    """Applies value operators to each instance in a stream based on specified fields.

    Args:
        operators_field (str): name of the field that contains a single name, or a list of names, of the operators to be applied,
            one after the other, for the processing of the instance. Each operator is equipped with 'process_instance()'
            method.

        default_operators (List[str]): A list of default operators to be used if no operators are found in the instance.

    Example:
        when instance {"prediction": 111, "references": [222, 333] , "c": ["processors.to_string", "processors.first_character"]}
        is processed by operator (please look up the catalog that these operators, they are tuned to process fields "prediction" and
        "references"):
        operator = ApplyOperatorsField(operators_field="c"),
        the resulting instance is: {"prediction": "1", "references": ["2", "3"], "c": ["processors.to_string", "processors.first_character"]}

    """

    operators_field: str
    default_operators: List[str] = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        operator_names = instance.get(self.operators_field)
        if operator_names is None:
            assert (
                self.default_operators is not None
            ), f"No operators found in field '{self.operators_field}', and no default operators provided."
            operator_names = self.default_operators

        if isinstance(operator_names, str):
            operator_names = [operator_names]
        # otherwise , operator_names is already a list

        # we now have a list of nanes of operators, each is equipped with process_instance method.
        operator = SequentialOperator(steps=operator_names)
        return operator.process_instance(instance, stream_name=stream_name)


class FilterByCondition(StreamOperator):
    """Filters a stream, yielding only instances in which the values in required fields follow the required condition operator.

    Raises an error if a required field name is missing from the input instance.

    Args:
       values (Dict[str, Any]): Field names and respective Values that instances must match according the condition, to be included in the output.

       condition: the name of the desired condition operator between the specified (sub) field's value  and the provided constant value.  Supported conditions are  ("gt", "ge", "lt", "le", "ne", "eq", "in","not in")

       error_on_filtered_all (bool, optional): If True, raises an error if all instances are filtered out. Defaults to True.

    Examples:
       | ``FilterByCondition(values = {"a":4}, condition = "gt")`` will yield only instances where field ``"a"`` contains a value ``> 4``
       | ``FilterByCondition(values = {"a":4}, condition = "le")`` will yield only instances where ``"a"<=4``
       | ``FilterByCondition(values = {"a":[4,8]}, condition = "in")`` will yield only instances where ``"a"`` is ``4`` or ``8``
       | ``FilterByCondition(values = {"a":[4,8]}, condition = "not in")`` will yield only instances where ``"a"`` is different from ``4`` or ``8``
       | ``FilterByCondition(values = {"a/b":[4,8]}, condition = "not in")`` will yield only instances where ``"a"`` is a dict in which key ``"b"`` is mapped to a value that is neither ``4`` nor ``8``
       | ``FilterByCondition(values = {"a[2]":4}, condition = "le")`` will yield only instances where "a" is a list whose 3-rd element is ``<= 4``


    """

    values: Dict[str, Any]
    condition: str
    condition_to_func = {
        "gt": operator.gt,
        "ge": operator.ge,
        "lt": operator.lt,
        "le": operator.le,
        "eq": operator.eq,
        "ne": operator.ne,
        "in": None,  # Handled as special case
        "not in": None,  # Handled as special case
    }
    error_on_filtered_all: bool = True

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        yielded = False
        for instance in stream:
            if self._is_required(instance):
                yielded = True
                yield instance

        if not yielded and self.error_on_filtered_all:
            raise RuntimeError(
                f"{self.__class__.__name__} filtered out every instance in stream '{stream_name}'. If this is intended set error_on_filtered_all=False"
            )

    def verify(self):
        if self.condition not in self.condition_to_func:
            raise ValueError(
                f"Unsupported condition operator '{self.condition}', supported {list(self.condition_to_func.keys())}"
            )

        for key, value in self.values.items():
            if self.condition in ["in", "not it"] and not isinstance(value, list):
                raise ValueError(
                    f"The filter for key ('{key}') in FilterByCondition with condition '{self.condition}' must be list but is not : '{value}'"
                )
        return super().verify()

    def _is_required(self, instance: dict) -> bool:
        for key, value in self.values.items():
            try:
                instance_key = dict_get(instance, key)
            except ValueError as ve:
                raise ValueError(
                    f"Required filter field ('{key}') in FilterByCondition is not found in instance."
                ) from ve
            if self.condition == "in":
                if instance_key not in value:
                    return False
            elif self.condition == "not in":
                if instance_key in value:
                    return False
            else:
                func = self.condition_to_func[self.condition]
                if func is None:
                    raise ValueError(
                        f"Function not defined for condition '{self.condition}'"
                    )
                if not func(instance_key, value):
                    return False
        return True


class FilterByConditionBasedOnFields(FilterByCondition):
    """Filters a stream based on a condition between 2 fields values.

    Raises an error if either of the required fields names is missing from the input instance.

    Args:
       values (Dict[str, str]): The fields names that the filter operation is based on.
       condition: the name of the desired condition operator between the specified field's values.  Supported conditions are  ("gt", "ge", "lt", "le", "ne", "eq", "in","not in")
       error_on_filtered_all (bool, optional): If True, raises an error if all instances are filtered out. Defaults to True.

    Examples:
       FilterByCondition(values = {"a":"b}, condition = "gt") will yield only instances where field "a" contains a value greater then the value in field "b".
       FilterByCondition(values = {"a":"b}, condition = "le") will yield only instances where "a"<="b"
    """

    def _is_required(self, instance: dict) -> bool:
        for key, value in self.values.items():
            try:
                instance_key = dict_get(instance, key)
            except ValueError as ve:
                raise ValueError(
                    f"Required filter field ('{key}') in FilterByCondition is not found in instance"
                ) from ve
            try:
                instance_value = dict_get(instance, value)
            except ValueError as ve:
                raise ValueError(
                    f"Required filter field ('{value}') in FilterByCondition is not found in instance"
                ) from ve
            if self.condition == "in":
                if instance_key not in instance_value:
                    return False
            elif self.condition == "not in":
                if instance_key in instance_value:
                    return False
            else:
                func = self.condition_to_func[self.condition]
                if func is None:
                    raise ValueError(
                        f"Function not defined for condition '{self.condition}'"
                    )
                if not func(instance_key, instance_value):
                    return False
        return True


class ComputeExpressionMixin(Artifact):
    """Computes an expression expressed over fields of an instance.

    Args:
        expression (str): the expression, in terms of names of fields of an instance
        imports_list (List[str]): list of names of imports needed for the evaluation of the expression
    """

    expression: str
    imports_list: List[str] = OptionalField(default_factory=list)

    def prepare(self):
        # can not do the imports here, because object does not pickle with imports
        self.globals = {
            module_name: __import__(module_name) for module_name in self.imports_list
        }

    def compute_expression(self, instance: dict) -> Any:
        if settings.allow_unverified_code:
            return eval(self.expression, {**self.globals, **instance})

        raise ValueError(
            f"Cannot evaluate expression in {self} when unitxt.settings.allow_unverified_code=False - either set it to True or set {settings.allow_unverified_code_key} environment variable."
            "\nNote: If using test_card() with the default setting, increase loader_limit to avoid missing conditions due to limited data sampling."
        )


class FilterByExpression(StreamOperator, ComputeExpressionMixin):
    """Filters a stream, yielding only instances which fulfil a condition specified as a string to be python's eval-uated.

    Raises an error if a field participating in the specified condition is missing from the instance

    Args:
        expression (str):
            a condition over fields of the instance, to be processed by python's eval()
        imports_list (List[str]):
            names of imports needed for the eval of the query (e.g. 're', 'json')
        error_on_filtered_all (bool, optional):
            If True, raises an error if all instances are filtered out. Defaults to True.

    Examples:
        | ``FilterByExpression(expression = "a > 4")`` will yield only instances where "a">4
        | ``FilterByExpression(expression = "a <= 4 and b > 5")`` will yield only instances where the value of field "a" is not exceeding 4 and in field "b" -- greater than 5
        | ``FilterByExpression(expression = "a in [4, 8]")`` will yield only instances where "a" is 4 or 8
        | ``FilterByExpression(expression = "a not in [4, 8]")`` will yield only instances where "a" is neither 4 nor 8
        | ``FilterByExpression(expression = "a['b'] not in [4, 8]")`` will yield only instances where "a" is a dict in which key 'b' is mapped to a value that is neither 4 nor 8
    """

    error_on_filtered_all: bool = True

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        yielded = False
        for instance in stream:
            if self.compute_expression(instance):
                yielded = True
                yield instance

        if not yielded and self.error_on_filtered_all:
            raise RuntimeError(
                f"{self.__class__.__name__} filtered out every instance in stream '{stream_name}'. If this is intended set error_on_filtered_all=False"
            )


class ExecuteExpression(InstanceOperator, ComputeExpressionMixin):
    """Compute an expression, specified as a string to be eval-uated, over the instance's fields, and store the result in field to_field.

    Raises an error if a field mentioned in the query is missing from the instance.

    Args:
       expression (str): an expression to be evaluated over the fields of the instance
       to_field (str): the field where the result is to be stored into
       imports_list (List[str]): names of imports needed for the eval of the query (e.g. 're', 'json')

    Examples:
       When instance {"a": 2, "b": 3} is process-ed by operator
       ExecuteExpression(expression="a+b", to_field = "c")
       the result is {"a": 2, "b": 3, "c": 5}

       When instance {"a": "hello", "b": "world"} is process-ed by operator
       ExecuteExpression(expression = "a+' '+b", to_field = "c")
       the result is {"a": "hello", "b": "world", "c": "hello world"}

    """

    to_field: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance[self.to_field] = self.compute_expression(instance)
        return instance


class ExtractMostCommonFieldValues(MultiStreamOperator):
    field: str
    stream_name: str
    overall_top_frequency_percent: Optional[int] = 100
    min_frequency_percent: Optional[int] = 0
    to_field: str
    process_every_value: Optional[bool] = False

    """
    Extract the unique values of a field ('field') of a given stream ('stream_name') and store (the most frequent of) them
    as a list in a new field ('to_field') in all streams.

    More specifically, sort all the unique values encountered in field 'field' by decreasing order of frequency.
    When 'overall_top_frequency_percent' is smaller than 100, trim the list from bottom, so that the total frequency of
    the remaining values makes 'overall_top_frequency_percent' of the total number of instances in the stream.
    When 'min_frequency_percent' is larger than 0, remove from the list any value whose relative frequency makes
    less than 'min_frequency_percent' of the total number of instances in the stream.
    At most one of 'overall_top_frequency_percent' and 'min_frequency_percent' is allowed to move from their default values.

    Examples:

    ExtractMostCommonFieldValues(stream_name="train", field="label", to_field="classes") - extracts all the unique values of
    field 'label', sorts them by decreasing frequency, and stores the resulting list in field 'classes' of each and
    every instance in all streams.

    ExtractMostCommonFieldValues(stream_name="train", field="labels", to_field="classes", process_every_value=True) -
    in case that field 'labels' contains a list of values (and not a single value) - track the occurrences of all the possible
    value members in these lists, and report the most frequent values.
    if process_every_value=False, track the most frequent whole lists, and report those (as a list of lists) in field
    'to_field' of each instance of all streams.

    ExtractMostCommonFieldValues(stream_name="train", field="label", to_field="classes",overall_top_frequency_percent=80) -
    extracts the most frequent possible values of field 'label' that together cover at least 80% of the instances of stream_name,
    and stores them in field 'classes' of each instance of all streams.

    ExtractMostCommonFieldValues(stream_name="train", field="label", to_field="classes",min_frequency_percent=5) -
    extracts all possible values of field 'label' that cover, each, at least 5% of the instances.
    Stores these values, sorted by decreasing order of frequency, in field 'classes' of each instance in all streams.
    """

    def verify(self):
        assert (
            self.overall_top_frequency_percent <= 100
            and self.overall_top_frequency_percent >= 0
        ), "'overall_top_frequency_percent' must be between 0 and 100"
        assert (
            self.min_frequency_percent <= 100 and self.min_frequency_percent >= 0
        ), "'min_frequency_percent' must be between 0 and 100"
        assert not (
            self.overall_top_frequency_percent < 100 and self.min_frequency_percent > 0
        ), "At most one of 'overall_top_frequency_percent' and 'min_frequency_percent' is allowed to move from their default value"
        super().verify()

    def process(self, multi_stream: MultiStream) -> MultiStream:
        stream = multi_stream[self.stream_name]
        counter = Counter()
        for instance in stream:
            if (not isinstance(instance[self.field], list)) and (
                self.process_every_value is True
            ):
                raise ValueError(
                    "'process_every_field' is allowed to change to 'True' only for fields whose contents are lists"
                )
            if (not isinstance(instance[self.field], list)) or (
                self.process_every_value is False
            ):
                # either not a list, or is a list but process_every_value == False : view contetns of 'field' as one entity whose occurrences are counted.
                counter.update(
                    [(*instance[self.field],)]
                    if isinstance(instance[self.field], list)
                    else [instance[self.field]]
                )  # convert to a tuple if list, to enable the use of Counter which would not accept
                # a list as an hashable entity to count its occurrences
            else:
                # content of 'field' is a list and process_every_value == True: add one occurrence on behalf of each individual value
                counter.update(instance[self.field])
        # here counter counts occurrences of individual values, or tuples.
        values_and_counts = counter.most_common()
        if self.overall_top_frequency_percent < 100:
            top_frequency = (
                sum(counter.values()) * self.overall_top_frequency_percent / 100.0
            )
            sum_counts = 0
            for _i, p in enumerate(values_and_counts):
                sum_counts += p[1]
                if sum_counts >= top_frequency:
                    break
            values_and_counts = counter.most_common(_i + 1)
        if self.min_frequency_percent > 0:
            min_frequency = self.min_frequency_percent * sum(counter.values()) / 100.0
            while values_and_counts[-1][1] < min_frequency:
                values_and_counts.pop()
        values_to_keep = [
            [*ele[0]] if isinstance(ele[0], tuple) else ele[0]
            for ele in values_and_counts
        ]

        addmostcommons = Set(fields={self.to_field: values_to_keep})
        return addmostcommons(multi_stream)


class ExtractFieldValues(ExtractMostCommonFieldValues):
    def verify(self):
        super().verify()

    def prepare(self):
        self.overall_top_frequency_percent = 100
        self.min_frequency_percent = 0


class Intersect(FieldOperator):
    """Intersects the value of a field, which must be a list, with a given list.

    Args:
        allowed_values (list) - list to intersect.
    """

    allowed_values: List[Any]

    def verify(self):
        super().verify()
        if self.process_every_value:
            raise ValueError(
                "'process_every_value=True' is not supported in Intersect operator"
            )

        if not isinstance(self.allowed_values, list):
            raise ValueError(
                f"The allowed_values is not a list but '{self.allowed_values}'"
            )

    def process_value(self, value: Any) -> Any:
        super().process_value(value)
        if not isinstance(value, list):
            raise ValueError(f"The value in field is not a list but '{value}'")
        return [e for e in value if e in self.allowed_values]


class IntersectCorrespondingFields(InstanceOperator):
    """Intersects the value of a field, which must be a list, with a given list , and removes corresponding elements from other list fields.

    For example:

    Assume the instances contain a field of 'labels' and a field with the labels' corresponding 'positions' in the text.

    .. code-block:: text

        IntersectCorrespondingFields(field="label",
                                    allowed_values=["b", "f"],
                                    corresponding_fields_to_intersect=["position"])

    would keep only "b" and "f" values in 'labels' field and
    their respective values in the 'position' field.
    (All other fields are not effected)

    .. code-block:: text

        Given this input:

        [
            {"label": ["a", "b"],"position": [0,1],"other" : "not"},
            {"label": ["a", "c", "d"], "position": [0,1,2], "other" : "relevant"},
            {"label": ["a", "b", "f"], "position": [0,1,2], "other" : "field"}
        ]

        So the output would be:
        [
                {"label": ["b"], "position":[1],"other" : "not"},
                {"label": [], "position": [], "other" : "relevant"},
                {"label": ["b", "f"],"position": [1,2], "other" : "field"},
        ]

    Args:
        field - the field to intersected (must contain list values)
        allowed_values (list) - list of values to keep
        corresponding_fields_to_intersect (list) - additional list fields from which values
        are removed based the corresponding indices of values removed from the 'field'
    """

    field: str
    allowed_values: List[str]
    corresponding_fields_to_intersect: List[str]

    def verify(self):
        super().verify()

        if not isinstance(self.allowed_values, list):
            raise ValueError(
                f"The allowed_values is not a type list but '{type(self.allowed_values)}'"
            )

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.field not in instance:
            raise ValueError(
                f"Field '{self.field}' is not in provided instance.\n"
                + to_pretty_string(instance)
            )

        for corresponding_field in self.corresponding_fields_to_intersect:
            if corresponding_field not in instance:
                raise ValueError(
                    f"Field '{corresponding_field}' is not in provided instance.\n"
                    + to_pretty_string(instance)
                )

        if not isinstance(instance[self.field], list):
            raise ValueError(
                f"Value of field '{self.field}' is not a list, so IntersectCorrespondingFields can not intersect with allowed values. Field value:\n"
                + to_pretty_string(instance, keys=[self.field])
            )

        num_values_in_field = len(instance[self.field])

        if set(self.allowed_values) == set(instance[self.field]):
            return instance

        indices_to_keep = [
            i
            for i, value in enumerate(instance[self.field])
            if value in set(self.allowed_values)
        ]

        result_instance = {}
        for field_name, field_value in instance.items():
            if (
                field_name in self.corresponding_fields_to_intersect
                or field_name == self.field
            ):
                if not isinstance(field_value, list):
                    raise ValueError(
                        f"Value of field '{field_name}' is not a list, IntersectCorrespondingFields can not intersect with allowed values."
                    )
                if len(field_value) != num_values_in_field:
                    raise ValueError(
                        f"Number of elements in field '{field_name}' is not the same as the number of elements in field '{self.field}' so the IntersectCorrespondingFields can not remove corresponding values.\n"
                        + to_pretty_string(instance, keys=[self.field, field_name])
                    )
                result_instance[field_name] = [
                    value
                    for index, value in enumerate(field_value)
                    if index in indices_to_keep
                ]
            else:
                result_instance[field_name] = field_value
        return result_instance


class RemoveValues(FieldOperator):
    """Removes elements in a field, which must be a list, using a given list of unallowed.

    Args:
        unallowed_values (list) - values to be removed.
    """

    unallowed_values: List[Any]

    def verify(self):
        super().verify()

        if not isinstance(self.unallowed_values, list):
            raise ValueError(
                f"The unallowed_values is not a list but '{self.unallowed_values}'"
            )

    def process_value(self, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError(f"The value in field is not a list but '{value}'")
        return [e for e in value if e not in self.unallowed_values]


class SplitByNestedGroup(MultiStreamOperator):
    """Splits a MultiStream that is small - for metrics, hence: whole stream can sit in memory, split by the value of field 'group'.

    Args:
        number_of_fusion_generations: int

    the value in field group is of the form "sourcen/sourcenminus1/..." describing the sources in which the instance sat
    when these were fused, potentially several phases of fusion. the name of the most recent source sits first in this value.
    (See BaseFusion and its extensions)
    number_of_fuaion_generations  specifies the length of the prefix by which to split the stream.
    E.g. for number_of_fusion_generations = 1, only the most recent fusion in creating this multi_stream, affects the splitting.
    For number_of_fusion_generations = -1, take the whole history written in this field, ignoring number of generations.
    """

    field_name_of_group: str = "group"
    number_of_fusion_generations: int = 1

    def process(self, multi_stream: MultiStream) -> MultiStream:
        result = defaultdict(list)

        for stream_name, stream in multi_stream.items():
            for instance in stream:
                if self.field_name_of_group not in instance:
                    raise ValueError(
                        f"Field {self.field_name_of_group} is missing from instance. Available fields: {instance.keys()}"
                    )
                signature = (
                    stream_name
                    + "~"  #  a sign that does not show within group values
                    + (
                        "/".join(
                            instance[self.field_name_of_group].split("/")[
                                : self.number_of_fusion_generations
                            ]
                        )
                        if self.number_of_fusion_generations >= 0
                        # for values with a smaller number of generations - take up to their last generation
                        else instance[self.field_name_of_group]
                        # for each instance - take all its generations
                    )
                )
                result[signature].append(instance)

        return MultiStream.from_iterables(result)


class AddIncrementalId(StreamOperator):
    to_field: str

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        for i, instance in enumerate(stream):
            instance[self.to_field] = i
            yield instance


class ApplyStreamOperatorsField(StreamOperator, ArtifactFetcherMixin):
    """Applies stream operators to a stream based on specified fields in each instance.

    Args:
        field (str): The field containing the operators to be applied.
        reversed (bool): Whether to apply the operators in reverse order.
    """

    field: str
    reversed: bool = False

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        first_instance = stream.peek()

        operators = first_instance.get(self.field, [])
        if isinstance(operators, str):
            operators = [operators]

        if self.reversed:
            operators = list(reversed(operators))

        for operator_name in operators:
            operator = self.get_artifact(operator_name)
            assert isinstance(
                operator, StreamingOperator
            ), f"Operator {operator_name} must be a StreamOperator"

            stream = operator(MultiStream({stream_name: stream}))[stream_name]

        yield from stream


def update_scores_of_stream_instances(stream: Stream, scores: List[dict]) -> Generator:
    for instance, score in zip(stream, scores):
        instance["score"] = recursive_copy(score)
        yield instance


class ApplyMetric(StreamOperator, ArtifactFetcherMixin):
    """Applies metric operators to a stream based on a metric field specified in each instance.

    Args:
        metric_field (str): The field containing the metrics to be applied.
        calc_confidence_intervals (bool): Whether the applied metric should calculate confidence intervals or not.
    """

    metric_field: str
    calc_confidence_intervals: bool

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        from .metrics import Metric, MetricsList

        # to be populated only when two or more metrics
        accumulated_scores = []

        first_instance = stream.peek()

        metric_names = first_instance.get(self.metric_field, [])
        if not metric_names:
            raise RuntimeError(
                f"Missing metric names in field '{self.metric_field}' and instance '{first_instance}'."
            )

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        metrics_list = []
        for metric_name in metric_names:
            metric = self.get_artifact(metric_name)
            if isinstance(metric, MetricsList):
                metrics_list.extend(list(metric.items))
            elif isinstance(metric, Metric):
                metrics_list.append(metric)
            else:
                raise ValueError(
                    f"Operator {metric_name} must be a Metric or MetricsList"
                )

        for metric in metrics_list:
            metric.set_confidence_interval_calculation(self.calc_confidence_intervals)
        # Each metric operator computes its score and then sets the main score, overwriting
        # the previous main score value (if any). So, we need to reverse the order of the listed metrics.
        # This will cause the first listed metric to run last, and the main score will be set
        # by the first listed metric (as desired).
        metrics_list = list(reversed(metrics_list))

        for i, metric in enumerate(metrics_list):
            if i == 0:  # first metric
                multi_stream = MultiStream({"tmp": stream})
            else:  # metrics with previous scores
                reusable_generator = ReusableGenerator(
                    generator=update_scores_of_stream_instances,
                    gen_kwargs={"stream": stream, "scores": accumulated_scores},
                )
                multi_stream = MultiStream.from_generators({"tmp": reusable_generator})

            multi_stream = metric(multi_stream)

            if i < len(metrics_list) - 1:  # last metric
                accumulated_scores = []
                for inst in multi_stream["tmp"]:
                    accumulated_scores.append(recursive_copy(inst["score"]))

        yield from multi_stream["tmp"]


class MergeStreams(MultiStreamOperator):
    """Merges multiple streams into a single stream.

    Args:
        new_stream_name (str): The name of the new stream resulting from the merge.
        add_origin_stream_name (bool): Whether to add the origin stream name to each instance.
        origin_stream_name_field_name (str): The field name for the origin stream name.
    """

    streams_to_merge: List[str] = None
    new_stream_name: str = "all"
    add_origin_stream_name: bool = True
    origin_stream_name_field_name: str = "origin"

    def merge(self, multi_stream) -> Generator:
        for stream_name, stream in multi_stream.items():
            if self.streams_to_merge is None or stream_name in self.streams_to_merge:
                for instance in stream:
                    if self.add_origin_stream_name:
                        instance[self.origin_stream_name_field_name] = stream_name
                    yield instance

    def process(self, multi_stream: MultiStream) -> MultiStream:
        return MultiStream(
            {
                self.new_stream_name: DynamicStream(
                    self.merge, gen_kwargs={"multi_stream": multi_stream}
                )
            }
        )


class Shuffle(PagedStreamOperator):
    """Shuffles the order of instances in each page of a stream.

    Args (of superclass):
        page_size (int): The size of each page in the stream. Defaults to 1000.
    """

    random_generator: Random = None

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        self.random_generator = new_random_generator(sub_seed="shuffle")

    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        self.random_generator.shuffle(page)
        yield from page


class FeatureGroupedShuffle(Shuffle):
    """Class for shuffling an input dataset by instance 'blocks', not on the individual instance level.

    Example is if the dataset consists of questions with paraphrases of it, and each question falls into a topic.
    All paraphrases have the same ID value as the original.
    In this case, we may want to shuffle on grouping_features = ['question ID'],
    to keep the paraphrases and original question together.
    We may also want to group by both 'question ID' and 'topic', if the question IDs are repeated between topics.
    In this case, grouping_features = ['question ID', 'topic']

    Args:
        grouping_features (list of strings): list of feature names to use to define the groups.
            a group is defined by each unique observed combination of data values for features in grouping_features
        shuffle_within_group (bool): whether to further shuffle the instances within each group block, keeping the block order

    Args (of superclass):
        page_size (int): The size of each page in the stream. Defaults to 1000.
            Note: shuffle_by_grouping_features determines the unique groups (unique combinations of values of grouping_features)
            separately by page (determined by page_size).  If a block of instances in the same group are split
            into separate pages (either by a page break falling in the group, or the dataset was not sorted by
            grouping_features), these instances will be shuffled separately and thus the grouping may be
            broken up by pages.  If the user wants to ensure the shuffle does the grouping and shuffling
            across all pages, set the page_size to be larger than the dataset size.
            See outputs_2features_bigpage and outputs_2features_smallpage in test_grouped_shuffle.
    """

    grouping_features: List[str] = None
    shuffle_within_group: bool = False

    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        if self.grouping_features is None:
            super().process(page, stream_name)
        else:
            yield from self.shuffle_by_grouping_features(page)

    def shuffle_by_grouping_features(self, page):
        import itertools
        from collections import defaultdict

        groups_to_instances = defaultdict(list)
        for item in page:
            groups_to_instances[
                tuple(item[ff] for ff in self.grouping_features)
            ].append(item)
        # now extract the groups (i.e., lists of dicts with order preserved)
        page_blocks = list(groups_to_instances.values())
        # and now shuffle the blocks
        self.random_generator.shuffle(page_blocks)
        if self.shuffle_within_group:
            blocks = []
            # reshuffle the instances within each block, but keep the blocks in order
            for block in page_blocks:
                self.random_generator.shuffle(block)
                blocks.append(block)
            page_blocks = blocks

        # now flatten the list so it consists of individual dicts, but in (randomized) block order
        return list(itertools.chain(*page_blocks))


class EncodeLabels(InstanceOperator):
    """Encode each value encountered in any field in 'fields' into the integers 0,1,...

    Encoding is determined by a str->int map that is built on the go, as different values are
    first encountered in the stream, either as list members or as values in single-value fields.

    Args:
        fields (List[str]): The fields to encode together.

    Example:
        applying ``EncodeLabels(fields = ["a", "b/*"])``
        on input stream = ``[{"a": "red", "b": ["red", "blue"], "c":"bread"},
        {"a": "blue", "b": ["green"], "c":"water"}]``   will yield the
        output stream = ``[{'a': 0, 'b': [0, 1], 'c': 'bread'}, {'a': 1, 'b': [2], 'c': 'water'}]``

        Note: dict_utils are applied here, and hence, fields that are lists, should be included in
        input 'fields' with the appendix ``"/*"``  as in the above example.

    """

    fields: List[str]

    def _process_multi_stream(self, multi_stream: MultiStream) -> MultiStream:
        self.encoder = {}
        return super()._process_multi_stream(multi_stream)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field_name in self.fields:
            values = dict_get(instance, field_name)
            values_was_a_list = isinstance(values, list)
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value not in self.encoder:
                    self.encoder[value] = len(self.encoder)
            new_values = [self.encoder[value] for value in values]
            if not values_was_a_list:
                new_values = new_values[0]
            dict_set(
                instance,
                field_name,
                new_values,
                not_exist_ok=False,  # the values to encode where just taken from there
                set_multiple="*" in field_name
                and isinstance(new_values, list)
                and len(new_values) > 0,
            )

        return instance


class StreamRefiner(StreamOperator):
    """Discard from the input stream all instances beyond the leading 'max_instances' instances.

    Thereby, if the input stream consists of no more than 'max_instances' instances, the resulting stream is the whole of the
    input stream. And if the input stream consists of more than 'max_instances' instances, the resulting stream only consists
    of the leading 'max_instances' of the input stream.

    Args:
        max_instances (int)
        apply_to_streams (optional, list(str)):
            names of streams to refine.

    Examples:
        when input = ``[{"a": 1},{"a": 2},{"a": 3},{"a": 4},{"a": 5},{"a": 6}]`` is fed into
        ``StreamRefiner(max_instances=4)``
        the resulting stream is ``[{"a": 1},{"a": 2},{"a": 3},{"a": 4}]``
    """

    max_instances: int = None
    apply_to_streams: Optional[List[str]] = None

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        if self.max_instances is not None:
            yield from stream.take(self.max_instances)
        else:
            yield from stream


class Deduplicate(StreamOperator):
    """Deduplicate the stream based on the given fields.

    Args:
        by (List[str]): A list of field names to deduplicate by. The combination of these fields' values will be used to determine uniqueness.

    Examples:
        >>> dedup = Deduplicate(by=["field1", "field2"])
    """

    by: List[str]

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        seen = set()

        for instance in stream:
            # Compute a lightweight hash for the signature
            signature = hash(str(tuple(dict_get(instance, field) for field in self.by)))

            if signature not in seen:
                seen.add(signature)
                yield instance


class Balance(StreamRefiner):
    """A class used to balance streams deterministically.

    For each instance, a signature is constructed from the values of the instance in specified input 'fields'.
    By discarding instances from the input stream, DeterministicBalancer maintains equal number of instances for all signatures.
    When also input 'max_instances' is specified, DeterministicBalancer maintains a total instance count not exceeding
    'max_instances'. The total number of discarded instances is as few as possible.

    Args:
        fields (List[str]):
            A list of field names to be used in producing the instance's signature.
        max_instances (Optional, int):
            overall max.

    Usage:
        ``balancer = DeterministicBalancer(fields=["field1", "field2"], max_instances=200)``
        ``balanced_stream = balancer.process(stream)``

    Example:
        When input ``[{"a": 1, "b": 1},{"a": 1, "b": 2},{"a": 2},{"a": 3},{"a": 4}]`` is fed into
        ``DeterministicBalancer(fields=["a"])``
        the resulting stream will be: ``[{"a": 1, "b": 1},{"a": 2},{"a": 3},{"a": 4}]``
    """

    fields: List[str]

    def signature(self, instance):
        return str(tuple(dict_get(instance, field) for field in self.fields))

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        counter = Counter()

        for instance in stream:
            counter[self.signature(instance)] += 1

        if len(counter) == 0:
            return

        lowest_count = counter.most_common()[-1][-1]

        max_total_instances_per_sign = lowest_count
        if self.max_instances is not None:
            max_total_instances_per_sign = min(
                lowest_count, self.max_instances // len(counter)
            )

        counter = Counter()

        for instance in stream:
            sign = self.signature(instance)
            if counter[sign] < max_total_instances_per_sign:
                counter[sign] += 1
                yield instance


class DeterministicBalancer(Balance):
    pass


class MinimumOneExamplePerLabelRefiner(StreamRefiner):
    """A class used to return a specified number instances ensuring at least one example  per label.

    For each instance, a signature value is constructed from the values of the instance in specified input ``fields``.
    ``MinimumOneExamplePerLabelRefiner`` takes first instance that appears from each label (each unique signature), and then adds more elements up to the max_instances limit.  In general, the refiner takes the first elements in the stream that meet the required conditions.
    ``MinimumOneExamplePerLabelRefiner`` then shuffles the results to avoid having one instance
    from each class first and then the rest . If max instance is not set, the original stream will be used

    Args:
        fields (List[str]):
            A list of field names to be used in producing the instance's signature.
        max_instances (Optional, int):
            Number of elements to select. Note that max_instances of StreamRefiners
            that are passed to the recipe (e.g. ``train_refiner``. ``test_refiner``) are overridden
            by the recipe parameters ( ``max_train_instances``, ``max_test_instances``)

    Usage:
        | ``balancer = MinimumOneExamplePerLabelRefiner(fields=["field1", "field2"], max_instances=200)``
        | ``balanced_stream = balancer.process(stream)``

    Example:
        When input ``[{"a": 1, "b": 1},{"a": 1, "b": 2},{"a": 1, "b": 3},{"a": 1, "b": 4},{"a": 2, "b": 5}]`` is fed into
        ``MinimumOneExamplePerLabelRefiner(fields=["a"], max_instances=3)``
        the resulting stream will be:
        ``[{'a': 1, 'b': 1}, {'a': 1, 'b': 2}, {'a': 2, 'b': 5}]`` (order may be different)
    """

    fields: List[str]

    def signature(self, instance):
        return str(tuple(dict_get(instance, field) for field in self.fields))

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        if self.max_instances is None:
            for instance in stream:
                yield instance

        counter = Counter()
        for instance in stream:
            counter[self.signature(instance)] += 1
        all_keys = counter.keys()
        if len(counter) == 0:
            return

        if self.max_instances is not None and len(all_keys) > self.max_instances:
            raise Exception(
                f"Can not generate a stream with at least one example per label, because the max instances requested  {self.max_instances} is smaller than the number of different labels {len(all_keys)}"
                f" ({len(all_keys)}"
            )

        counter = Counter()
        used_indices = set()
        selected_elements = []
        # select at least one per class
        for idx, instance in enumerate(stream):
            sign = self.signature(instance)
            if counter[sign] == 0:
                counter[sign] += 1
                used_indices.add(idx)
                selected_elements.append(
                    instance
                )  # collect all elements first to allow shuffling of both groups

        # select more to reach self.max_instances examples
        for idx, instance in enumerate(stream):
            if idx not in used_indices:
                if self.max_instances is None or len(used_indices) < self.max_instances:
                    used_indices.add(idx)
                    selected_elements.append(
                        instance
                    )  # collect all elements first to allow shuffling of both groups

        # shuffle elements to avoid having one element from each class appear first
        random_generator = new_random_generator(sub_seed=selected_elements)
        random_generator.shuffle(selected_elements)
        yield from selected_elements


class LengthBalancer(DeterministicBalancer):
    """Balances by a signature that reflects the total length of the fields' values, quantized into integer segments.

    Args:
        segments_boundaries (List[int]):
            distinct integers sorted in increasing order, that map a given total length
            into the index of the least of them that exceeds the given total length.
            (If none exceeds -- into one index beyond, namely, the length of segments_boundaries)
        fields (Optional, List[str]):
            the total length of the values of these fields goes through the quantization described above


    Example:
        when input ``[{"a": [1, 3], "b": 0, "id": 0}, {"a": [1, 3], "b": 0, "id": 1}, {"a": [], "b": "a", "id": 2}]``
        is fed into ``LengthBalancer(fields=["a"], segments_boundaries=[1])``,
        input instances will be counted and balanced against two categories:
        empty total length (less than 1), and non-empty.
    """

    segments_boundaries: List[int]
    fields: Optional[List[str]]

    def signature(self, instance):
        total_len = 0
        for field_name in self.fields:
            total_len += len(dict_get(instance, field_name))
        for i, val in enumerate(self.segments_boundaries):
            if total_len < val:
                return i
        return i + 1


class DownloadError(Exception):
    def __init__(
        self,
        message,
    ):
        self.__super__(message)


class UnexpectedHttpCodeError(Exception):
    def __init__(self, http_code):
        self.__super__(f"unexpected http code {http_code}")


class DownloadOperator(SideEffectOperator):
    """Operator for downloading a file from a given URL to a specified local path.

    Args:
        source (str):
            URL of the file to be downloaded.
        target (str):
            Local path where the downloaded file should be saved.
    """

    source: str
    target: str

    def process(self):
        try:
            response = requests.get(self.source, allow_redirects=True)
        except Exception as e:
            raise DownloadError(f"Unabled to download {self.source}") from e
        if response.status_code != 200:
            raise UnexpectedHttpCodeError(response.status_code)
        with open(self.target, "wb") as f:
            f.write(response.content)


class ExtractZipFile(SideEffectOperator):
    """Operator for extracting files from a zip archive.

    Args:
        zip_file (str):
            Path of the zip file to be extracted.
        target_dir (str):
            Directory where the contents of the zip file will be extracted.
    """

    zip_file: str
    target_dir: str

    def process(self):
        with zipfile.ZipFile(self.zip_file) as zf:
            zf.extractall(self.target_dir)


class DuplicateInstances(StreamOperator):
    """Operator which duplicates each instance in stream a given number of times.

    Args:
        num_duplications (int):
            How many times each instance should be duplicated (1 means no duplication).
        duplication_index_field (Optional[str]):
            If given, then additional field with specified name is added to each duplicated instance,
            which contains id of a given duplication. Defaults to None, so no field is added.
    """

    num_duplications: int
    duplication_index_field: Optional[str] = None

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        for instance in stream:
            for idx in range(self.num_duplications):
                duplicate = recursive_shallow_copy(instance)
                if self.duplication_index_field:
                    duplicate.update({self.duplication_index_field: idx})
                yield duplicate

    def verify(self):
        if not isinstance(self.num_duplications, int) or self.num_duplications < 1:
            raise ValueError(
                f"num_duplications must be an integer equal to or greater than 1. "
                f"Got: {self.num_duplications}."
            )

        if self.duplication_index_field is not None and not isinstance(
            self.duplication_index_field, str
        ):
            raise ValueError(
                f"If given, duplication_index_field must be a string. "
                f"Got: {self.duplication_index_field}"
            )


class CollateInstances(StreamOperator):
    """Operator which collates values from multiple instances to a single instance.

    Each field becomes the list of values of corresponding field of collated `batch_size` of instances.

    Attributes:
        batch_size (int)

    Example:
        .. code-block:: text

            CollateInstances(batch_size=2)

            Given inputs = [
                {"a": 1, "b": 2},
                {"a": 2, "b": 2},
                {"a": 3, "b": 2},
                {"a": 4, "b": 2},
                {"a": 5, "b": 2}
            ]

            Returns targets = [
                {"a": [1,2], "b": [2,2]},
                {"a": [3,4], "b": [2,2]},
                {"a": [5], "b": [2]},
            ]


    """

    batch_size: int

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        stream = list(stream)
        for i in range(0, len(stream), self.batch_size):
            batch = stream[i : i + self.batch_size]
            new_instance = {}
            for a_field in batch[0]:
                if a_field == "data_classification_policy":
                    flattened_list = [
                        classification
                        for instance in batch
                        for classification in instance[a_field]
                    ]
                    new_instance[a_field] = sorted(set(flattened_list))
                else:
                    new_instance[a_field] = [instance[a_field] for instance in batch]
            yield new_instance

    def verify(self):
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(
                f"batch_size must be an integer equal to or greater than 1. "
                f"Got: {self.batch_size}."
            )


class CollateInstancesByField(StreamOperator):
    """Groups a list of instances by a specified field, aggregates specified fields into lists, and ensures consistency for all other non-aggregated fields.

    Args:
        by_field str: the name of the field to group data by.
        aggregate_fields list(str): the field names to aggregate into lists.

    Returns:
        A stream of instances grouped and aggregated by the specified field.

    Raises:
        UnitxtError: If non-aggregate fields have inconsistent values.

    Example:
        Collate the instances based on field "category" and aggregate fields "value" and "id".

        .. code-block:: text

            CollateInstancesByField(by_field="category", aggregate_fields=["value", "id"])

            given input:
            [
                {"id": 1, "category": "A", "value": 10", "flag" : True},
                {"id": 2, "category": "B", "value": 20", "flag" : False},
                {"id": 3, "category": "A", "value": 30", "flag" : True},
                {"id": 4, "category": "B", "value": 40", "flag" : False}
            ]

            the output is:
            [
                {"category": "A", "id": [1, 3], "value": [10, 30], "info": True},
                {"category": "B", "id": [2, 4], "value": [20, 40], "info": False}
            ]

        Note that the "flag" field is not aggregated, and must be the same
        in all instances in the same category, or an error is raised.
    """

    by_field: str = NonPositionalField(required=True)
    aggregate_fields: List[str] = NonPositionalField(required=True)

    def prepare(self):
        super().prepare()

    def verify(self):
        super().verify()
        if not isinstance(self.by_field, str):
            raise UnitxtError(
                f"The 'by_field' value is not a string but '{type(self.by_field)}'"
            )

        if not isinstance(self.aggregate_fields, list):
            raise UnitxtError(
                f"The 'allowed_field_values' is not a list but '{type(self.aggregate_fields)}'"
            )

    def process(self, stream: Stream, stream_name: Optional[str] = None):
        grouped_data = {}

        for instance in stream:
            if self.by_field not in instance:
                raise UnitxtError(
                    f"The field '{self.by_field}' specified by CollateInstancesByField's 'by_field' argument is not found in instance."
                )
            for k in self.aggregate_fields:
                if k not in instance:
                    raise UnitxtError(
                        f"The field '{k}' specified in CollateInstancesByField's 'aggregate_fields' argument is not found in instance."
                    )
            key = instance[self.by_field]

            if key not in grouped_data:
                grouped_data[key] = {
                    k: v for k, v in instance.items() if k not in self.aggregate_fields
                }
                # Add empty lists for fields to aggregate
                for agg_field in self.aggregate_fields:
                    if agg_field in instance:
                        grouped_data[key][agg_field] = []

            for k, v in instance.items():
                # Merge classification policy list across instance with same key
                if k == "data_classification_policy" and instance[k]:
                    grouped_data[key][k] = sorted(set(grouped_data[key][k] + v))
                # Check consistency for all non-aggregate fields
                elif k != self.by_field and k not in self.aggregate_fields:
                    if k in grouped_data[key] and grouped_data[key][k] != v:
                        raise ValueError(
                            f"Inconsistent value for field '{k}' in group '{key}': "
                            f"'{grouped_data[key][k]}' vs '{v}'. Ensure that all non-aggregated fields in CollateInstancesByField are consistent across all instances."
                        )
                # Aggregate fields
                elif k in self.aggregate_fields:
                    grouped_data[key][k].append(instance[k])

        yield from grouped_data.values()


class WikipediaFetcher(FieldOperator):
    mode: Literal["summary", "text"] = "text"
    _requirements_list = ["Wikipedia-API"]

    def prepare(self):
        super().prepare()
        import wikipediaapi

        self.wikipedia = wikipediaapi.Wikipedia("Unitxt")

    def process_value(self, value: Any) -> Any:
        title = value.split("/")[-1]
        page = self.wikipedia.page(title)

        return {"title": page.title, "body": getattr(page, self.mode)}


class Fillna(FieldOperator):
    value: Any

    def process_value(self, value: Any) -> Any:
        import numpy as np

        try:
            if np.isnan(value):
                return self.value
        except TypeError:
            return value
        return value
