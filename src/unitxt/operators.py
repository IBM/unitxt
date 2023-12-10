import collections
import importlib
import os
import uuid
from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import field
from itertools import zip_longest
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Union,
)

from .artifact import Artifact, fetch_artifact
from .dataclass import NonPositionalField
from .dict_utils import dict_delete, dict_get, dict_set, is_subpath
from .operator import (
    MultiStream,
    MultiStreamOperator,
    PagedStreamOperator,
    SingleStreamOperator,
    SingleStreamReducer,
    StreamingOperator,
    StreamInitializerOperator,
    StreamInstanceOperator,
    StreamSource,
)
from .random_utils import get_random, nested_seed
from .stream import Stream
from .text_utils import nested_tuple_to_string
from .utils import flatten_dict


class FromIterables(StreamInitializerOperator):
    """Creates a MultiStream from a dict of named iterables.

    Example:
        operator = FromIterables()
        ms = operator.process(iterables)

    """

    def process(self, iterables: Dict[str, Iterable]) -> MultiStream:
        return MultiStream.from_iterables(iterables)


class IterableSource(StreamSource):
    """Creates a MultiStream from a dict of named iterables.

    It is a callable.

    Args:
        iterables (Dict[str, Iterable]): A dictionary mapping stream names to iterables.

    Example:
        operator =  IterableSource(input_dict)
        ms = operator()

    """

    iterables: Dict[str, Iterable]

    def __call__(self) -> MultiStream:
        return MultiStream.from_iterables(self.iterables)


class MapInstanceValues(StreamInstanceOperator):
    """A class used to map instance values into a stream.

    This class is a type of StreamInstanceOperator,
    it maps values of instances in a stream using predefined mappers.

    Attributes:
        mappers (Dict[str, Dict[str, str]]): The mappers to use for mapping instance values.
            Keys are the names of the fields to be mapped, and values are dictionaries
            that define the mapping from old values to new values.
        strict (bool): If True, the mapping is applied strictly. That means if a value
            does not exist in the mapper, it will raise a KeyError. If False, values
            that are not present in the mapper are kept as they are.
        process_every_value (bool): If True, all fields to be mapped should be lists, and the mapping
            is to be applied to their individual elements. If False, mapping is only applied to a field
            containing a single value.

    Examples:
        MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}})
        replaces '1' with 'hi' and '2' with 'bye' in field 'a' in all instances of all streams:
        instance {"a":"1", "b": 2} becomes {"a":"hi", "b": 2}.

        MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}, process_every_element=True)
        Assuming field 'a' is a list of values, potentially including "1"-s and "2"-s, this replaces
        each such "1" with "hi" and "2" -- with "bye" in all instances of all streams:
        instance {"a": ["1", "2"], "b": 2} becomes {"a": ["hi", "bye"], "b": 2}.

        MapInstanceValues(mappers={"a": {"1": "hi", "2": "bye"}}, strict=True)
        To ensure that all values of field 'a' are mapped in every instance, use strict=True.
        Input instance {"a":"3", "b": 2} will raise an exception per the above call,
        because "3" is not a key in the mapper of "a".

        MapInstanceValues(mappers={"a": {str([1,2,3,4]): 'All', str([]): 'None'}}, strict=True)
        replaces a list [1,2,3,4] with the string 'All' and an empty list by string 'None'.
        Note that mapped values are defined by their string representation, so mapped values
        must be converted to strings.
    """

    mappers: Dict[str, Dict[str, str]]
    strict: bool = True
    use_query: bool = False
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
            value = dict_get(instance, key, use_dpath=self.use_query)
            if value is not None:
                if (self.process_every_value is True) and (not isinstance(value, list)):
                    raise ValueError(
                        f"'process_every_field' == True is allowed only when all fields which have mappers, i.e., {list(self.mappers.keys())} are lists. Instace = {instance}"
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
                    use_dpath=self.use_query,
                )

        return instance

    def get_mapped_value(self, instance, key, mapper, val):
        val_as_str = str(val)  # make sure the value is a string
        if self.strict and (val_as_str not in mapper):
            raise KeyError(
                f"value '{val}' in instance '{instance}' is not found in mapper '{mapper}', associated with field '{key}'."
            )
        # By default deep copy the value in mapper to avoid shared modifications
        if val_as_str in mapper:
            return deepcopy(mapper[val_as_str])
        return val


class FlattenInstances(StreamInstanceOperator):
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


class AddFields(StreamInstanceOperator):
    """Adds specified fields to each instance in a given stream or all streams (default) If fields exist, updates them.

    Args:
        fields (Dict[str, object]): The fields to add to each instance.
        use_query (bool) : Use '/' to access inner fields
        use_deepcopy (bool) : Deep copy the input value to avoid later modifications

    Examples:
        # Add a 'classes' field with a value of a list "positive" and "negative" to all streams
        AddFields(fields={"classes": ["positive","negatives"]})

        # Add a 'start' field under the 'span' field with a value of 0 to all streams
        AddFields(fields={"span/start": 0}

        # Add a 'classes' field with a value of a list "positive" and "negative" to 'train' stream
        AddFields(fields={"classes": ["positive","negatives"], apply_to_stream=["train"]})

        # Add a 'classes' field on a given list, prevent modification of original list
        # from changing the instance.
        AddFields(fields={"classes": alist}), use_deepcopy=True)
        # if now alist is modified, still the instances remain intact.
    """

    fields: Dict[str, object]
    use_query: bool = False
    use_deepcopy: bool = False

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.use_query:
            for key, value in self.fields.items():
                if self.use_deepcopy:
                    value = deepcopy(value)
                dict_set(instance, key, value, use_dpath=self.use_query)
        else:
            if self.use_deepcopy:
                self.fields = deepcopy(self.fields)
            instance.update(self.fields)
        return instance


class RemoveFields(StreamInstanceOperator):
    """Remove specified fields to each instance in a stream.

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


class FieldOperator(StreamInstanceOperator):
    """A general stream instance operator that processes the values of a field (or multiple ones).

    Args:
        field (Optional[str]): The field to process, if only a single one is passed Defaults to None
        to_field (Optional[str]): Field name to save, if only one field is to be saved, if None is passed the operation would happen in-place and replace "field". Defaults to None
        field_to_field (Optional[Union[List[List[str]], Dict[str, str]]]): Mapping from fields to process to their names after this process,
         duplicates are allowed. Inner List, if used, should be of length 2. Defaults to None
        process_every_value (bool): Processes the values in a list instead of the list as a value, similar to *var. Defaults to False
        use_query (bool): Whether to use dpath style queries. Defaults to False.

        Note: if 'field' and 'to_field' (or both members of a pair in 'field_to_field') are equal (or share a common
        prefix if 'use_query'=True), then the result of the operation is saved within 'field'
    """

    field: Optional[str] = None
    to_field: Optional[str] = None
    field_to_field: Optional[Union[List[List[str]], Dict[str, str]]] = None
    process_every_value: bool = False
    use_query: bool = False
    get_default: Any = None
    not_exist_ok: bool = False

    def verify(self):
        super().verify()

        assert (
            self.field is not None or self.field_to_field is not None
        ), "Must supply a field to work on"
        assert (
            self.to_field is None or self.field_to_field is None
        ), f"Can not apply operator to create both on {self.to_field} and on the mapping from fields to fields {self.field_to_field}"
        assert (
            self.field is None or self.field_to_field is None
        ), f"Can not apply operator both on {self.field} and on the from fields in the mapping {self.field_to_field}"
        assert self._field_to_field, f"the from and to fields must be defined or implied from the other inputs got: {self._field_to_field}"
        # self._field_to_field is built explicitly by pairs, or copied from argument 'field_to_field'
        for pair in self._field_to_field:
            assert (
                len(pair) == 2
            ), f"when 'field_to_field' is defined as a list of lists, the inner lists should all be of length 2. {self.field_to_field}"

    @abstractmethod
    def process_value(self, value: Any) -> Any:
        pass

    def prepare(self):
        super().prepare()

        # prepare is invoked before verify, hence must make some checks here, before the changes done here
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

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for from_field, to_field in self._field_to_field:
            try:
                old_value = dict_get(
                    instance,
                    from_field,
                    use_dpath=self.use_query,
                    default=self.get_default,
                    not_exist_ok=self.not_exist_ok,
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to get '{from_field}' from {instance} due to : {e}"
                ) from e
            try:
                if self.process_every_value:
                    new_value = [self.process_value(value) for value in old_value]
                else:
                    new_value = self.process_value(old_value)
            except Exception as e:
                raise ValueError(
                    f"Failed to process '{from_field}' from {instance} due to : {e}"
                ) from e
            if is_subpath(from_field, to_field) or is_subpath(to_field, from_field):
                dict_delete(instance, from_field)
            dict_set(
                instance,
                to_field,
                new_value,
                use_dpath=self.use_query,
                not_exist_ok=True,
            )
        return instance


class RenameFields(FieldOperator):
    """Renames fields.

    Move value from one field to another, potentially, if 'use_query'=True, from one branch into another.
    Remove the from field, potentially part of it in case of use_query.

    Examples:
        RenameFields(field_to_field={"b": "c"})
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "c": 2}, {"a": 2, "c": 3}]

        RenameFields(field_to_field={"b": "c/d"}, use_query=True)
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "c": {"d": 2}}, {"a": 2, "c": {"d": 3}}]

        RenameFields(field_to_field={"b": "b/d"}, use_query=True)
        will change inputs [{"a": 1, "b": 2}, {"a": 2, "b": 3}] to [{"a": 1, "b": {"d": 2}}, {"a": 2, "b": {"d": 3}}]

        RenameFields(field_to_field={"b/c/e": "b/d"}, use_query=True)
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
                dict_delete(res, from_field)
                if self.use_query:
                    from_field_components = list(
                        os.path.normpath(from_field).split(os.path.sep)
                    )
                    while len(from_field_components) > 1:
                        from_field_components.pop()
                        parent = dict_get(res, os.path.sep.join(from_field_components))
                        if isinstance(parent, dict) and not parent:
                            dict_delete(res, os.path.sep.join(from_field_components))
                        else:
                            break

        return res


class AddConstant(FieldOperator):
    """Adds a constant, being argument 'add', to the processed value.

    Args:
        add: the constant to add.
    """

    add: Any

    def process_value(self, value: Any) -> Any:
        return self.add + value


class Augmentor(StreamInstanceOperator):
    """A stream that augments the values of either the task input fields before rendering with the template,  or the  input passed to the model after rendering of the template.

    Args:
        augment_model_input: Whether to augment the input to the model.
        augment_task_input:  Whether to augment the task input fields.  The specific fields are defined in the FormTask operator.

    """

    augment_task_input: bool = False
    augment_model_input: bool = False

    def verify(self):
        assert not (
            self.augment_task_input and self.augment_model_input
        ), "Augmentor must set either 'augment_task_input' and 'augment_model_input' but not both"
        assert (
            self.augment_task_input or self.augment_model_input
        ), "Augmentor must set either 'augment_task_input' or 'augment_model_input'"

        super().verify()

    @abstractmethod
    def process_value(self, value: Any) -> Any:
        pass

    def prepare(self):
        pass

    def set_task_input_fields(self, task_input_fields: List[str]):
        self._task_input_fields = [
            "inputs/" + task_input_field for task_input_field in task_input_fields
        ]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.augment_task_input:
            assert (
                len(self._task_input_fields) > 0
            ), "No augmentable input fields were defined in FormTask, and augmentation was requested. Specify the fields to augment in 'argumentable_inputs' attribute of the FormTask."
            fields = self._task_input_fields
            assert not self.augment_model_input

        if self.augment_model_input:
            fields = ["source"]
            assert not self.augment_task_input

        for field_name in fields:
            try:
                old_value = dict_get(
                    instance,
                    field_name,
                    use_dpath=True,
                    default="",
                    not_exist_ok=False,
                )
            except ValueError as e:
                raise TypeError(f"Failed to get {field_name} from {instance}") from e

            # We are setting a nested seed based on the value processed, to ensure that
            # the augmentation randomizations do not effect other randomization choices and
            # to make the augmentation randomization choices different for each text.
            with nested_seed(str(hash(old_value))):
                try:
                    new_value = self.process_value(old_value)
                except Exception as e:
                    raise RuntimeError(
                        f"Error augmenting value '{old_value}' from '{field_name}' in instance: {instance}"
                    ) from e
            dict_set(instance, field_name, new_value, use_dpath=True, not_exist_ok=True)
        return instance


class NullAugmentor(Augmentor):
    def verify(self):
        pass

    def process_value(self, value: Any) -> Any:
        return value


class AugmentWhitespace(Augmentor):
    """Augments the inputs by replace existing whitespace with other whitespace.

    Currently each whitespace is replaced by a random choice of 1-3 whitespace charaters (spcae, tab, newline).
    """

    def process_value(self, value: Any) -> Any:
        import re

        words = re.split(r"(\s+)", value)
        new_value = ""

        for word in words:
            if word.isspace():
                new_value += get_random().choice(
                    ["\n", "\t", " "]
                ) * get_random().randint(1, 3)
            else:
                new_value += word
        return new_value


class AugmentSuffix(Augmentor):
    r"""Augments the input by appending to it a randomly selected (typically, whitespace) pattern.

    Args:
     suffixes : the potential (typically, whitespace) patterns to select from.
        The dictionary version allows to specify relative weights of the different patterns.
     remove_existing_trailing_whitespaces : allows to first clean existing trailing whitespaces.
        The selected pattern is then appended to the potentially trimmed at its end input.


    Examples:
        to append a '\n' or a '\t' to the end of the input, employ
        AugmentSuffix(augment_model_input=True, suffixes=['\n','\t'])
        If '\n' is preferred over '\t', at 2:1 ratio, employ
        AugmentSuffix(augment_model_input=True, suffixes={'\n':2,'\t':1})
        which will append '\n' twice as often as '\t'.

    """

    suffixes: Optional[Union[List[str], Dict[str, int]]] = [" ", "\n", "\t"]
    remove_existing_trailing_whitespaces: Optional[bool] = False

    def verify(self):
        assert (
            isinstance(self.suffixes, list) or isinstance(self.suffixes, dict)
        ), f"Argument 'suffixes' should be either a list or a dictionary, whereas it is of type {type(self.suffixes)}"

        if isinstance(self.suffixes, dict):
            for k, v in self.suffixes.items():
                assert isinstance(
                    k, str
                ), f"suffixes should map strings, whereas key {k!s} is of type {type(k)}"
                assert isinstance(
                    v, int
                ), f"suffixes should map to ints, whereas value {v!s} is of type {type(v)}"
        else:
            for k in self.suffixes:
                assert isinstance(
                    k, str
                ), f"suffixes should be a list of strings, whereas member {k!s} is of type {type(k)}"
        super().verify()

    def prepare(self):
        self.pats = (
            self.suffixes
            if isinstance(self.suffixes, list)
            else [k for k, v in self.suffixes.items()]
        )
        total_weight = (
            len(self.pats)
            if isinstance(self.suffixes, list)
            else sum([v for k, v in self.suffixes.items()])
        )
        self.weights = (
            [1.0 / total_weight] * len(self.pats)
            if isinstance(self.suffixes, list)
            else [float(self.suffixes[p]) / total_weight for p in self.pats]
        )

        super().prepare()

    def process_value(self, value: Any) -> Any:
        assert value is not None, "input value should not be None"
        new_value = str(value)
        if self.remove_existing_trailing_whitespaces:
            new_value = new_value.rstrip()
        new_value += get_random().choices(self.pats, self.weights, k=1)[0]

        return new_value


class ShuffleFieldValues(FieldOperator):
    """Shuffles a list of values found in a field."""

    def process_value(self, value: Any) -> Any:
        res = list(value)
        get_random().shuffle(res)
        return res


class JoinStr(FieldOperator):
    """Joins a list of strings (contents of a field), similar to str.join().

    Args:
        separator (str): text to put between values
    """

    separator: str = ","

    def process_value(self, value: Any) -> Any:
        return self.separator.join(str(x) for x in value)


class Apply(StreamInstanceOperator):
    """A class used to apply a python function and store the result in a field.

    Args:
        function (str): name of function.
        to_field (str): the field to store the result
        additional arguments are field names passed to the function

    Examples:
    Store in field  "b" the uppercase string of the value in field "a"
    Apply("a", function=str.upper, to_field="b")

    Dump the json representation of field "t" and store back in the same field.
    Apply("t", function=json.dumps, to_field="t")

    Set the time in a field 'b'.
    Apply(function=time.time, to_field="b")

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
        splitted = function_str.split(".", 1)
        if len(splitted) == 1:
            return __builtins__[splitted[0]]

        module_name, function_name = splitted
        if module_name in __builtins__:
            obj = __builtins__[module_name]
        elif module_name in globals():
            obj = globals()[module_name]
        else:
            obj = importlib.import_module(module_name)
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


class ListFieldValues(StreamInstanceOperator):
    """Concatenates values of multiple fields into a list, and assigns it to a new field."""

    fields: List[str]
    to_field: str
    use_query: bool = False

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        values = []
        for field_name in self.fields:
            values.append(dict_get(instance, field_name, use_dpath=self.use_query))
        instance[self.to_field] = values
        return instance


class ZipFieldValues(StreamInstanceOperator):
    """Zips values of multiple fields in a given instance, similar to list(zip(*fields)).

    The value in each of the specified 'fields' is assumed to be a list. The lists from all 'fields'
    are zipped, and stored into 'to_field'.

    If 'longest'=False, the length of the zipped result is determined by the shortest input value.
    If 'longest'=False, the length of the zipped result is determined by the longest input, padding shorter
    inputs with None -s.

    """

    fields: List[str]
    to_field: str
    longest: bool = False
    use_query: bool = False

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        values = []
        for field_name in self.fields:
            values.append(dict_get(instance, field_name, use_dpath=self.use_query))
        if self.longest:
            zipped = zip_longest(*values)
        else:
            zipped = zip(*values)
        instance[self.to_field] = list(zipped)
        return instance


class IndexOf(StreamInstanceOperator):
    """For a given instance, finds the offset of value of field 'index_of', within the value of field 'search_in'."""

    search_in: str
    index_of: str
    to_field: str
    use_query: bool = False

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        lst = dict_get(instance, self.search_in, use_dpath=self.use_query)
        item = dict_get(instance, self.index_of, use_dpath=self.use_query)
        instance[self.to_field] = lst.index(item)
        return instance


class TakeByField(StreamInstanceOperator):
    """From field 'field' of a given instance, select the member indexed by field 'index', and store to field 'to_field'."""

    field: str
    index: str
    to_field: str = None
    use_query: bool = False

    def prepare(self):
        if self.to_field is None:
            self.to_field = self.field

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        value = dict_get(instance, self.field, use_dpath=self.use_query)
        index_value = dict_get(instance, self.index, use_dpath=self.use_query)
        instance[self.to_field] = value[index_value]
        return instance


class CopyFields(FieldOperator):
    """Copies specified fields from one field to another.

    Args:
        field_to_field (Union[List[List], Dict[str, str]]): A list of lists, where each sublist contains the source field and the destination field, or a dictionary mapping source fields to destination fields.
        use_dpath (bool): Whether to use dpath for accessing fields. Defaults to False.
    """

    def process_value(self, value: Any) -> Any:
        return value


class AddID(StreamInstanceOperator):
    id_field_name: str = "id"

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance[self.id_field_name] = str(uuid.uuid4()).replace("-", "")
        return instance


class CastFields(StreamInstanceOperator):
    """Casts specified fields to specified types.

    Args:
        types (Dict[str, str]): A dictionary mapping fields to their new types.
        nested (bool): Whether to cast nested fields. Defaults to False.
        fields (Dict[str, str]): A dictionary mapping fields to their new types.
        defaults (Dict[str, object]): A dictionary mapping types to their default values for cases of casting failure.
    """

    types = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }
    fields: Dict[str, str] = field(default_factory=dict)
    failure_defaults: Dict[str, object] = field(default_factory=dict)
    use_nested_query: bool = False
    cast_multiple: bool = False

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
        values = [self._cast_single(value, type, field) for value in values]

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field_name, type in self.fields.items():
            value = dict_get(instance, field_name, use_dpath=self.use_nested_query)
            if self.cast_multiple:
                casted_value = self._cast_multiple(value, type, field_name)
            else:
                casted_value = self._cast_single(value, type, field_name)
            dict_set(
                instance, field_name, casted_value, use_dpath=self.use_nested_query
            )
        return instance


def recursive_divide(instance, divisor, strict=False):
    if isinstance(instance, dict):
        for key, value in instance.items():
            instance[key] = recursive_divide(value, divisor, strict=strict)
    elif isinstance(instance, list):
        for i, value in enumerate(instance):
            instance[i] = recursive_divide(value, divisor, strict=strict)
    elif isinstance(instance, float):
        instance /= divisor
    elif strict:
        raise ValueError(f"Cannot divide instance of type {type(instance)}")
    return instance


class DivideAllFieldsBy(StreamInstanceOperator):
    divisor: float = 1.0
    strict: bool = False
    recursive: bool = True

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return recursive_divide(instance, self.divisor, strict=self.strict)


class ArtifactFetcherMixin:
    """Provides a way to fetch and cache artifacts in the system.

    Args:
        cache (Dict[str, Artifact]): A cache for storing fetched artifacts.
    """

    cache: Dict[str, Artifact] = {}

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if artifact_identifier not in cls.cache:
            artifact, artifactory = fetch_artifact(artifact_identifier)
            cls.cache[artifact_identifier] = artifact
        return cls.cache[artifact_identifier]


class ApplyOperatorsField(StreamInstanceOperator, ArtifactFetcherMixin):
    """Applies value operators to each instance in a stream based on specified fields.

    Args:
        value_field (str): The field containing the value to be operated on.
        operators_field (str): The field containing the operators to be applied.
        default_operators (List[str]): A list of default operators to be used if no operators are found in the instance.
    """

    inputs_fields: str

    operators_field: str
    default_operators: List[str] = None
    fields_to_treat_as_list: List[str] = NonPositionalField(default_factory=list)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        operator_names = instance.get(self.operators_field)
        if operator_names is None:
            assert (
                self.default_operators is not None
            ), f"No operators found in {self.field} field and no default operators provided"
            operator_names = self.default_operators

        if isinstance(operator_names, str):
            operator_names = [operator_names]

        for name in operator_names:
            operator = self.get_artifact(name)
            for field_name in self.inputs_fields:
                value = instance[field_name]
                if field_name in self.fields_to_treat_as_list:
                    instance[field_name] = [operator.process(v) for v in value]
                else:
                    instance[field_name] = operator.process(instance[field_name])

        return instance


class FilterByValues(SingleStreamOperator):
    """Filters a stream, yielding only instances that match specified values in the provided fields.

    Args:
        values (Dict[str, Any]): For each field, the values that instances should match to be included in the output.
    """

    required_values: Dict[str, Any]

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        for instance in stream:
            filter = False
            for key, value in self.required_values.items():
                if key not in instance:
                    raise ValueError(
                        f"Required filter field ('{key}') in FilterByValues is not found in {instance}"
                    )
                if instance[key] != value:
                    filter = True
            if not filter:
                yield instance


class ExtractFieldValues(MultiStreamOperator):
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

    ExtractFieldValues(stream_name="train", field="label", to_field="classes") - extracts all the unique values of
    field 'label', sorts them by decreasing frequency, and stores the resulting list in field 'classes' of each and
    every instance in all streams.

    ExtractFieldValues(stream_name="train", field="labels", to_field="classes", process_every_value=True) -
    in case that field 'labels' contains a list of values (and not a single value) - track the occurrences of all the possible
    value members in these lists, and report the most frequent values.
    if process_every_value=False, track the most frequent whole lists, and report those (as a list of lists) in field
    'to_field' of each instance of all streams.

    ExtractFieldValues(stream_name="train", field="label", to_field="classes",overall_top_frequency_percent=80) -
    extracts the most frequent possible values of field 'label' that together cover at least 80% of the instances of stream_name,
    and stores them in field 'classes' of each instance of all streams.

    ExtractFieldValues(stream_name="train", field="label", to_field="classes",min_frequency_percent=5) -
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
        all_values = []
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
                all_values.append(
                    (*instance[self.field],)
                    if isinstance(instance[self.field], list)
                    else instance[self.field]
                )  # convert to a tuple if list, to enable the use of Counter which would not accept
                # a list as an entity to count its occurrences
            else:
                # content of 'field' is a list and process_every_value == True: add one occurrence on behalf of each individual value
                all_values.extend(instance[self.field])
        counter = Counter(
            all_values
        )  # here all_values is a list of individual values, or tupples. Hence, Counter is feasible
        values_and_counts = counter.most_common()
        if self.overall_top_frequency_percent < 100:
            top_frequency = len(all_values) * self.overall_top_frequency_percent / 100.0
            sum_counts = 0
            for _i, p in enumerate(values_and_counts):
                sum_counts += p[1]
                if sum_counts >= top_frequency:
                    break
            values_and_counts = counter.most_common(_i + 1)
        if self.min_frequency_percent > 0:
            min_frequency = self.min_frequency_percent * len(all_values) / 100.0
            while values_and_counts[-1][1] < min_frequency:
                values_and_counts.pop()
        values_to_keep = [
            [*ele[0]] if isinstance(ele[0], tuple) else ele[0]
            for ele in values_and_counts
        ]
        for name in multi_stream:
            for instance in multi_stream[name]:
                instance[self.to_field] = values_to_keep
        return multi_stream


class FilterByListsOfValues(SingleStreamOperator):
    """Filters a stream, yielding only instances that  whose field values are included in the specified value lists.

    Args:
        required_values (Dict[str, List]): For each field, the list of values that instances should match to be included in the output.
    """

    required_values: Dict[str, List]
    error_on_filtered_all: bool = True

    def verify(self):
        super().verify()
        for key, value in self.required_values.items():
            if not isinstance(value, list):
                raise ValueError(
                    f"The filter for key ('{key}') in FilterByListsOfValues is not a list but '{value}'"
                )

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        filtered_all = True
        for instance in stream:
            filter = False
            for key, value in self.required_values.items():
                if key not in instance:
                    raise ValueError(
                        f"Required filter field ('{key}') in FilterByListsOfValues is not found in {instance}"
                    )
                if instance[key] not in value:
                    filter = True
            if not filter:
                filtered_all = False
                yield instance
        if filtered_all and self.error_on_filtered_all:
            raise RuntimeError(
                f"FilterByListsOfValues filtered out every instance in stream '{stream_name}'. If this is intended set error_on_filtered_all=False"
            )


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


class RemoveValues(FieldOperator):
    """Removes elements in a field, which must be a list, using a given list of unallowed.

    Args:
        unallowed_values (list) - removed_values.
    """

    unallowed_values: List[Any]

    def verify(self):
        super().verify()
        if self.process_every_value:
            raise ValueError(
                "'process_every_value=True' is not supported in RemoveValues operator"
            )

        if not isinstance(self.unallowed_values, list):
            raise ValueError(
                f"The unallowed_values is not a list but '{self.unallowed_values}'"
            )

    def process_value(self, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError(f"The value in field is not a list but '{value}'")
        return [e for e in value if e not in self.unallowed_values]


class Unique(SingleStreamReducer):
    """Reduces a stream to unique instances based on specified fields.

    Args:
        fields (List[str]): The fields that should be unique in each instance.
    """

    fields: List[str] = field(default_factory=list)

    @staticmethod
    def to_tuple(instance: dict, fields: List[str]) -> tuple:
        result = []
        for field_name in fields:
            value = instance[field_name]
            if isinstance(value, list):
                value = tuple(value)
            result.append(value)
        return tuple(result)

    def process(self, stream: Stream) -> Stream:
        seen = set()
        for instance in stream:
            values = self.to_tuple(instance, self.fields)
            if values not in seen:
                seen.add(values)
        return list(seen)


class SplitByValue(MultiStreamOperator):
    """Splits a MultiStream into multiple streams based on unique values in specified fields.

    Args:
        fields (List[str]): The fields to use when splitting the MultiStream.
    """

    fields: List[str] = field(default_factory=list)

    def process(self, multi_stream: MultiStream) -> MultiStream:
        uniques = Unique(fields=self.fields)(multi_stream)

        result = {}

        for stream_name, stream in multi_stream.items():
            stream_unique_values = uniques[stream_name]
            for unique_values in stream_unique_values:
                filtering_values = dict(zip(self.fields, unique_values))
                filtered_streams = FilterByValues(
                    required_values=filtering_values
                )._process_single_stream(stream)
                filtered_stream_name = (
                    stream_name + "_" + nested_tuple_to_string(unique_values)
                )
                result[filtered_stream_name] = filtered_streams

        return MultiStream(result)


class ApplyStreamOperatorsField(SingleStreamOperator, ArtifactFetcherMixin):
    """Applies stream operators to a stream based on specified fields in each instance.

    Args:
        field (str): The field containing the operators to be applied.
        reversed (bool): Whether to apply the operators in reverse order.
    """

    field: str
    reversed: bool = False

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        first_instance = stream.peak()

        operators = first_instance.get(self.field, [])
        if isinstance(operators, str):
            operators = [operators]

        if self.reversed:
            operators = list(reversed(operators))

        for operator_name in operators:
            operator = self.get_artifact(operator_name)
            assert isinstance(
                operator, StreamingOperator
            ), f"Operator {operator_name} must be a SingleStreamOperator"

            stream = operator(MultiStream({"tmp": stream}))["tmp"]

        yield from stream


class ApplyMetric(SingleStreamOperator, ArtifactFetcherMixin):
    """Applies metric operators to a stream based on a metric field specified in each instance.

    Args:
        metric_field (str): The field containing the metrics to be applied.
        calc_confidence_intervals (bool): Whether the applied metric should calculate confidence intervals or not.
    """

    metric_field: str
    calc_confidence_intervals: bool

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        from .metrics import Metric, MetricPipeline, MetricWithConfidenceInterval

        first_instance = stream.peak()

        metric_names = first_instance.get(self.metric_field, [])
        if not metric_names:
            raise RuntimeError(
                f"Missing metric names in field '{self.metric_field}' and instance '{first_instance}'."
            )

        if isinstance(metric_names, str):
            metric_names = [metric_names]

        # Each metric operator computes its score and then sets the main score, overwriting
        # the previous main score value (if any). So, we need to reverse the order of the listed metrics.
        # This will cause the first listed metric to run last, and the main score will be set
        # by the first listed metric (as desired).
        metric_names = list(reversed(metric_names))

        for metric_name in metric_names:
            metric = self.get_artifact(metric_name)
            assert isinstance(
                metric, Metric
            ), f"Operator {metric_name} must be a Metric"

            if not self.calc_confidence_intervals:
                if isinstance(metric, MetricWithConfidenceInterval):
                    metric.disable_confidence_interval_calculation()
                elif isinstance(metric, MetricPipeline) and isinstance(
                    metric.metric, MetricWithConfidenceInterval
                ):
                    metric.metric.disable_confidence_interval_calculation()

            stream = metric(MultiStream({"tmp": stream}))["tmp"]

        yield from stream


class AddFieldNamePrefix(StreamInstanceOperator):
    """Adds a prefix to each field name in each instance of a stream.

    Args:
        prefix_dict (Dict[str, str]): A dictionary mapping stream names to prefixes.
    """

    prefix_dict: Dict[str, str]

    def prepare(self):
        return super().prepare()

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return {
            self.prefix_dict[stream_name] + key: value
            for key, value in instance.items()
        }


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

    def merge(self, multi_stream):
        for stream_name, stream in multi_stream.items():
            if self.streams_to_merge is None or stream_name in self.streams_to_merge:
                for instance in stream:
                    if self.add_origin_stream_name:
                        instance[self.origin_stream_name_field_name] = stream_name
                    yield instance

    def process(self, multi_stream: MultiStream) -> MultiStream:
        return MultiStream(
            {
                self.new_stream_name: Stream(
                    self.merge, gen_kwargs={"multi_stream": multi_stream}
                )
            }
        )


class Shuffle(PagedStreamOperator):
    """Shuffles the order of instances in each page of a stream.

    Args (of superclass):
        page_size (int): The size of each page in the stream. Defaults to 1000.
    """

    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        get_random().shuffle(page)
        yield from page


class EncodeLabels(StreamInstanceOperator):
    """Encode each value encountered in any field in 'fields' into the integers 0,1,...

    Encoding is determined by a str->int map that is built on the go, as different values are
    first encountered in the stream, either as list members or as values in single-value fields.

    Args:
        fields (List[str]): The fields to encode together.

    Example: applying
        EncodeLabels(fields = ["a", "b/*"])
        on input stream = [{"a": "red", "b": ["red", "blue"], "c":"bread"},
          {"a": "blue", "b": ["green"], "c":"water"}]   will yield the
        output stream = [{'a': 0, 'b': [0, 1], 'c': 'bread'}, {'a': 1, 'b': [2], 'c': 'water'}]

        Note: dpath is applied here, and hence, fields that are lists, should be included in
        input 'fields' with the appendix "/*"  as in the above example.

    """

    fields: List[str]

    def _process_multi_stream(self, multi_stream: MultiStream) -> MultiStream:
        self.encoder = {}
        return super()._process_multi_stream(multi_stream)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        for field_name in self.fields:
            values = dict_get(instance, field_name, use_dpath=True)
            if not isinstance(values, list):
                values = [values]
            for value in values:
                if value not in self.encoder:
                    self.encoder[value] = len(self.encoder)
            new_values = [self.encoder[value] for value in values]
            dict_set(
                instance, field_name, new_values, use_dpath=True, set_multiple=True
            )

        return instance


class StreamRefiner(SingleStreamOperator):
    max_instances: int = None

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        if self.max_instances is not None:
            yield from stream.take(self.max_instances)
        else:
            yield from stream


class DeterministicBalancer(StreamRefiner):
    """A class used to balance streams deterministically.

    Attributes:
        fields (List[str]): A list of field names to be used in determining the signature of an instance.
        streams (List[str]): A list of stream names to be processed by the balancer.

    Usage:
        balancer = DeterministicBalancer(fields=["field1", "field2"], streams=["stream1", "stream2"])
        balanced_stream = balancer.process(stream)
    """

    fields: List[str]

    def signature(self, instance):
        return str(
            tuple(dict_get(instance, field, use_dpath=True) for field in self.fields)
        )

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        counter = collections.Counter()

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

        counter = collections.Counter()

        for instance in stream:
            sign = self.signature(instance)
            if counter[sign] < max_total_instances_per_sign:
                counter[sign] += 1
                yield instance


class LengthBalancer(DeterministicBalancer):
    segments_boundaries: List[int]

    def signature(self, instance):
        total_len = 0
        for field_name in self.fields:
            total_len += len(dict_get(instance, field_name, use_dpath=True))
        for i, val in enumerate(self.segments_boundaries):
            if total_len < val:
                return i
        return i + 1
