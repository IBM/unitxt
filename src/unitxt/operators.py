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

General or Specelized Operators
--------------------------------
Some operators are specielized in specific task such as:

- :class:`loaders<unitxt.loaders>` for loading data.
- :class:`splitters<unitxt.splitters>` for fixing data splits.

Other specelized operators are used by unitxt internally:

- :class:`templates<unitxt.templates>` for verbalizing data examples.
- :class:`formats<unitxt.formats>` for preparing data for models.

The rest of this section is dedicated for general operators.

General Operaotrs List:
------------------------
"""
import collections
import importlib
import operator
import os
import uuid
import zipfile
from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from dataclasses import field
from importlib import import_module
from itertools import zip_longest
from random import Random
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import requests

from .artifact import Artifact, fetch_artifact
from .dataclass import NonPositionalField, OptionalField
from .dict_utils import dict_delete, dict_get, dict_set, is_subpath
from .operator import (
    MultiStream,
    MultiStreamOperator,
    PagedStreamOperator,
    SequentialOperator,
    SideEffectOperator,
    SingleStreamOperator,
    SingleStreamReducer,
    SourceOperator,
    StreamingOperator,
    StreamInitializerOperator,
    StreamInstanceOperator,
)
from .random_utils import new_random_generator
from .settings_utils import get_settings
from .stream import Stream
from .text_utils import nested_tuple_to_string
from .type_utils import isoftype
from .utils import flatten_dict

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


class MapInstanceValues(StreamInstanceOperator):
    """A class used to map instance values into other values.

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


class FieldOperator(StreamInstanceOperator):
    """A general stream instance operator that processes the values of a field (or multiple ones).

    Args:
        field (Optional[str]): The field to process, if only a single one is passed. Defaults to None
        to_field (Optional[str]): Field name to save result into, if only one field is processed, if None is passed the
          operation would happen in-place and its result would replace the value of "field". Defaults to None
        field_to_field (Optional[Union[List[List[str]], Dict[str, str]]]): Mapping from names of fields to process,
          to names of fields to save the results into. Inner List, if used, should be of length 2.
          A field is processed by feeding its value into method 'process_value' and storing the result in to_field that
          is mapped to the field.
          When the type of argument 'field_to_field' is List, the order by which the fields are processed is their order
          in the (outer) List. But when the type of argument 'field_to_field' is Dict, there is no uniquely determined
          order. The end result might depend on that order if either (1) two different fields are mapped to the same
          to_field, or (2) a field shows both as a key and as a value in different mappings.
          The operator throws an AssertionError in either of these cases.
          field_to_field defaults to None
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
        assert (
            len(self._field_to_field) > 0
        ), f"'input argument 'field_to_field' should convey at least one field to process. Got {self.field_to_field}"
        # self._field_to_field is built explicitly by pairs, or copied from argument 'field_to_field'
        if self.field_to_field is None:
            return
        # for backward compatibility also allow list of tupples of two strings
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
    """A stream operator that augments the values of either the task input fields before rendering with the template,  or the input passed to the model after rendering of the template.

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

            try:
                new_value = self.process_value(old_value)
            except Exception as e:
                raise RuntimeError(
                    f"Error augmenting value '{old_value}' from '{field_name}' in instance: {instance}"
                ) from e
            dict_set(instance, field_name, new_value, use_dpath=True, not_exist_ok=True)
        return instance


class NullAugmentor(Augmentor):
    """Does not change the input string."""

    def verify(self):
        pass

    def process_value(self, value: Any) -> Any:
        return value


class AugmentWhitespace(Augmentor):
    """Augments the inputs by replacing existing whitespaces with other whitespaces.

    Currently, each whitespace is replaced by a random choice of 1-3 whitespace characters (space, tab, newline).
    """

    def process_value(self, value: Any) -> Any:
        import re

        words = re.split(r"(\s+)", value)
        new_value = ""

        random_generator = new_random_generator(sub_seed=value)
        for word in words:
            if word.isspace():
                new_value += random_generator.choice(
                    ["\n", "\t", " "]
                ) * random_generator.randint(1, 3)
            else:
                new_value += word
        return new_value


class AugmentPrefixSuffix(Augmentor):
    r"""Augments the input by prepending and appending to it a randomly selected (typically, whitespace) patterns.

    Args:
     prefixes, suffixes (list or dict) : the potential (typically, whitespace) patterns to select from.
        The dictionary version allows to specify relative weights of the different patterns.
     prefix_len, suffix_len (positive int) : The added prefix or suffix will be of length
        prefix_len of suffix_len, respectively, repetitions of the randomly selected patterns.
     remove_existing_whitespaces : allows to first clean any existing leading and trailing whitespaces.
        The strings made of repetitions of the selected pattern(s) are then prepended and/or appended to the potentially
        trimmed input.
     If only one of prefixes/suffixes is needed, set the other to None.

    Examples:
        To prepend the input with a prefix made of 4 '\n'-s or '\t'-s, employ
        AugmentPrefixSuffix(augment_model_input=True, prefixes=['\n','\t'], prefix_len=4, suffixes = None)
        To append the input with a suffix made of 3 '\n'-s or '\t'-s, with triple '\n' suffixes
        being preferred over triple '\t', at 2:1 ratio, employ
        AugmentPrefixSuffix(augment_model_input=True, suffixes={'\n':2,'\t':1}, suffix_len=3, prefixes = None)
        which will append '\n'-s twice as often as '\t'-s.

    """

    prefixes: Optional[Union[List[str], Dict[str, int]]] = {
        " ": 20,
        "\\t": 10,
        "\\n": 40,
        "": 30,
    }
    prefix_len: Optional[int] = 3
    suffixes: Optional[Union[List[str], Dict[str, int]]] = {
        " ": 20,
        "\\t": 10,
        "\\n": 40,
        "": 30,
    }
    suffix_len: Optional[int] = 3
    remove_existing_whitespaces: Optional[bool] = False

    def verify(self):
        assert (
            self.prefixes or self.suffixes
        ), "At least one of prefixes/suffixes should be not None."
        for arg, arg_name in zip(
            [self.prefixes, self.suffixes], ["prefixes", "suffixes"]
        ):
            assert (
                arg is None or isoftype(arg, List[str]) or isoftype(arg, Dict[str, int])
            ), f"Argument {arg_name} should be either None or a list of strings or a dictionary str->int. {arg} is none of the above."
        assert (
            self.prefix_len > 0
        ), f"prefix_len must be positive, got {self.prefix_len}"
        assert (
            self.suffix_len > 0
        ), f"suffix_len must be positive, got {self.suffix_len}"
        super().verify()

    def _calculate_distributions(self, prefs_or_suffs):
        if prefs_or_suffs is None:
            return None, None
        patterns = (
            prefs_or_suffs
            if isinstance(prefs_or_suffs, list)
            else [k for k, v in prefs_or_suffs.items()]
        )
        total_weight = (
            len(patterns)
            if isinstance(prefs_or_suffs, list)
            else sum([v for k, v in prefs_or_suffs.items()])
        )
        weights = (
            [1.0 / total_weight] * len(patterns)
            if isinstance(prefs_or_suffs, list)
            else [float(prefs_or_suffs[p]) / total_weight for p in patterns]
        )
        return patterns, weights

    def prepare(self):
        # Being an artifact, prepare is invoked before verify. Here we need verify before the actions
        self.verify()
        self._prefix_pattern_distribution = {"length": self.prefix_len}
        self._suffix_pattern_distribution = {"length": self.suffix_len}

        (
            self._prefix_pattern_distribution["patterns"],
            self._prefix_pattern_distribution["weights"],
        ) = self._calculate_distributions(self.prefixes)
        (
            self._suffix_pattern_distribution["patterns"],
            self._suffix_pattern_distribution["weights"],
        ) = self._calculate_distributions(self.suffixes)
        super().prepare()

    def _get_random_pattern(
        self, pattern_distribution, random_generator: Random
    ) -> str:
        string_to_add = ""
        if pattern_distribution["patterns"]:
            string_to_add = "".join(
                random_generator.choices(
                    pattern_distribution["patterns"],
                    pattern_distribution["weights"],
                    k=pattern_distribution["length"],
                )
            )
        return string_to_add

    def process_value(self, value: Any) -> Any:
        assert value is not None, "input value should not be None"
        new_value = str(value)
        if self.remove_existing_whitespaces:
            new_value = new_value.strip()
        random_generator = new_random_generator(sub_seed=value)
        prefix = self._get_random_pattern(
            self._prefix_pattern_distribution, random_generator
        )
        suffix = self._get_random_pattern(
            self._suffix_pattern_distribution, random_generator
        )
        return prefix + new_value + suffix


class ShuffleFieldValues(FieldOperator):
    """Shuffles a list of values found in a field."""

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


class Perturbate(FieldOperator):
    """Slightly perturbates the contents of 'field'. Could be Handy for imitating prediction from given target.

    When task was classification, argument 'select_from' can be used to list the other potential classes, as a
    relevant perturbation
    """

    select_from: List[Any] = []
    percentage_to_perturbate: int = 1  # 1 percent

    def verify(self):
        assert (
            0 <= self.percentage_to_perturbate and self.percentage_to_perturbate <= 100
        ), f"'percentage_to_perturbate' should be in the range 0..100. Received {self.percentage_to_perturbate}"

    def prepare(self):
        super().prepare()
        self.random_generator = new_random_generator(sub_seed="CopyWithPerturbation")

    def process_value(self, value: Any) -> Any:
        perturbate = (
            self.random_generator.randint(1, 100) <= self.percentage_to_perturbate
        )
        if not perturbate:
            return value

        if value in self.select_from:
            # 80% of cases, return a decent class, otherwise, perturbate the value itself as follows
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


class CopyFields(FieldOperator):
    """Copies values from specified fields to specified fields.

    Args (of parent class):
        field_to_field (Union[List[List], Dict[str, str]]): A list of lists, where each sublist contains the source field and the destination field, or a dictionary mapping source fields to destination fields.
        use_query (bool): Whether to use dpath for accessing fields. Defaults to False.

    Examples:
        An input instance {"a": 2, "b": 3}, when processed by
        CopyField(field_to_field={"a": "b"}
        would yield {"a": 2, "b": 2}, and when processed by
        CopyField(field_to_field={"a": "c"} would yield
        {"a": 2, "b": 3, "c": 2}

        with use_query=True, we can also copy inside the field:
        CopyFields(field_to_field={"a/0": "a"}, use_query=True)
        would process instance {"a": [1, 3]} into {"a": 1}


    """

    def process_value(self, value: Any) -> Any:
        return value


class AddID(StreamInstanceOperator):
    """Stores a unique id value in the designated 'id_field_name' field of the given instance."""

    id_field_name: str = "id"

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        instance[self.id_field_name] = str(uuid.uuid4()).replace("-", "")
        return instance


class CastFields(StreamInstanceOperator):
    """Casts specified fields to specified types.

    Args:
        use_nested_query (bool): Whether to cast nested fields, expressed in dpath. Defaults to False.
        fields (Dict[str, str]): A dictionary mapping field names to the names of the types to cast the fields to.
            e.g: "int", "str", "float", "bool". Basic names of types
        defaults (Dict[str, object]): A dictionary mapping field names to default values for cases of casting failure.
        process_every_value (bool): If true, all fields involved must contain lists, and each value in the list is then casted. Defaults to False.

    Examples:
        CastFields(
                fields={"a/d": "float", "b": "int"},
                failure_defaults={"a/d": 0.0, "b": 0},
                process_every_value=True,
                use_nested_query=True
            )
        would process the input instance: {"a": {"d": ["half", "0.6", 1, 12]}, "b": ["2"]}
            into {"a": {"d": [0.0, 0.6, 1.0, 12.0]}, "b": [2]}

    """

    fields: Dict[str, str] = field(default_factory=dict)
    failure_defaults: Dict[str, object] = field(default_factory=dict)
    use_nested_query: bool = False
    process_every_value: bool = False

    def prepare(self):
        self.types = {"int": int, "float": float, "str": str, "bool": bool}

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
            value = dict_get(instance, field_name, use_dpath=self.use_nested_query)
            if self.process_every_value:
                assert isinstance(
                    value, list
                ), f"'process_every_value' can be set to True only for fields that contain lists, whereas in instance {instance}, the contents of field '{field_name}' is of type '{type(value)}'"
                casted_value = self._cast_multiple(value, type, field_name)
            else:
                casted_value = self._cast_single(value, type, field_name)
            dict_set(
                instance, field_name, casted_value, use_dpath=self.use_nested_query
            )
        return instance


class DivideAllFieldsBy(StreamInstanceOperator):
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

    cache: Dict[str, Artifact] = {}

    @classmethod
    def get_artifact(cls, artifact_identifier: str) -> Artifact:
        if artifact_identifier not in cls.cache:
            artifact, artifactory = fetch_artifact(artifact_identifier)
            cls.cache[artifact_identifier] = artifact
        return cls.cache[artifact_identifier]


class ApplyOperatorsField(StreamInstanceOperator):
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
        return operator.process_instance(instance)


class FilterByCondition(SingleStreamOperator):
    """Filters a stream, yielding only instances for which the required values follows the required condition operator.

    Raises an error if a required key is missing.

    Args:
       values (Dict[str, Any]): Values that instances must match using the condition to be included in the output.
       condition: the name of the desired condition operator between the key and the value in values ("gt", "ge", "lt", "le", "ne", "eq")
       error_on_filtered_all (bool, optional): If True, raises an error if all instances are filtered out. Defaults to True.

    Examples:
       FilterByCondition(values = {"a":4}, condition = "gt") will yield only instances where "a">4
       FilterByCondition(values = {"a":4}, condition = "le") will yield only instances where "a"<=4
       FilterByCondition(values = {"a":[4,8]}, condition = "in") will yield only instances where "a" is 4 or 8
       FilterByCondition(values = {"a":[4,8]}, condition = "not in") will yield only instances where "a" different from 4 or 8

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
            if key not in instance:
                raise ValueError(
                    f"Required filter field ('{key}') in FilterByCondition is not found in {instance}"
                )
            if self.condition == "in":
                if instance[key] not in value:
                    return False
            elif self.condition == "not in":
                if instance[key] in value:
                    return False
            else:
                func = self.condition_to_func[self.condition]
                if func is None:
                    raise ValueError(
                        f"Function not defined for condition '{self.condition}'"
                    )
                if not func(instance[key], value):
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
        self.globs = {}
        self.to_import = True

    def compute_expression(self, instance: dict) -> Any:
        if self.to_import:
            for module_name in self.imports_list:
                self.globs[module_name] = import_module(module_name)
            self.to_import = False

        if settings.allow_unverified_code:
            return eval(self.expression, self.globs, instance)

        raise ValueError(
            f"Cannot run expression by {self} when unitxt.settings.allow_unverified_code=False either set it to True or set {settings.allow_unverified_code_key} environment variable."
        )


class FilterByExpression(SingleStreamOperator, ComputeExpressionMixin):
    """Filters a stream, yielding only instances which fulfil a condition specified as a string to be python's eval-uated.

    Raises an error if a field participating in the specified condition is missing from the instance

    Args:
       expression (str): a condition over fields of the instance, to be processed by python's eval()
       imports_list (List[str]): names of imports needed for the eval of the query (e.g. 're', 'json')
       error_on_filtered_all (bool, optional): If True, raises an error if all instances are filtered out. Defaults to True.

    Examples:
       FilterByExpression(expression = "a > 4") will yield only instances where "a">4
       FilterByExpression(expression = "a <= 4 and b > 5") will yield only instances where the value of field "a" is not exceeding 4 and in field "b" -- greater than 5
       FilterByExpression(expression = "a in [4, 8]") will yield only instances where "a" is 4 or 8
       FilterByExpression(expression = "a not in [4, 8]") will yield only instances where "a" is neither 4 nor 8

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


class ExecuteExpression(StreamInstanceOperator, ComputeExpressionMixin):
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
        # here counter counts occurrences of individual values, or tupples.
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

        addmostcommons = AddFields(fields={self.to_field: values_to_keep})
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


class RemoveValues(FieldOperator):
    """Removes elements in a field, which must be a list, using a given list of unallowed.

    Args:
        unallowed_values (list) - values to be removed.
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
                filtered_streams = FilterByCondition(
                    values=filtering_values, condition="eq"
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

        first_instance = stream.peek()

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

    random_generator: Random = None

    def before_process_multi_stream(self):
        super().before_process_multi_stream()
        self.random_generator = new_random_generator(sub_seed="shuffle")

    def process(self, page: List[Dict], stream_name: Optional[str] = None) -> Generator:
        self.random_generator.shuffle(page)
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
    """Discard from the input stream all instances beyond the leading 'max_instances' instances.

    Thereby, if the input stream consists of no more than 'max_instances' instances, the resulting stream is the whole of the
    input stream. And if the input stream consists of more than 'max_instances' instances, the resulting stream only consists
    of the leading 'max_instances' of the input stream.

    Args:  max_instances (int)
           apply_to_streams (optional, list(str)): names of streams to refine.

    Examples:
        when input = [{"a": 1},{"a": 2},{"a": 3},{"a": 4},{"a": 5},{"a": 6}] is fed into
        StreamRefiner(max_instances=4)
        the resulting stream is [{"a": 1},{"a": 2},{"a": 3},{"a": 4}]
    """

    max_instances: int = None
    apply_to_streams: Optional[List[str]] = None

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        if self.max_instances is not None:
            yield from stream.take(self.max_instances)
        else:
            yield from stream


class DeterministicBalancer(StreamRefiner):
    """A class used to balance streams deterministically.

    For each instance, a signature is constructed from the values of the instance in specified input 'fields'.
    By discarding instances from the input stream, DeterministicBalancer maintains equal number of instances for all signatures.
    When also input 'max_instances' is specified, DeterministicBalancer maintains a total instance count not exceeding
    'max_instances'. The total number of discarded instances is as few as possible.

    Attributes:
        fields (List[str]): A list of field names to be used in producing the instance's signature.
        max_instances (Optional, int)

    Usage:
        balancer = DeterministicBalancer(fields=["field1", "field2"], max_instances=200)
        balanced_stream = balancer.process(stream)

    Example:
        When input [{"a": 1, "b": 1},{"a": 1, "b": 2},{"a": 2},{"a": 3},{"a": 4}] is fed into
        DeterministicBalancer(fields=["a"])
        the resulting stream will be: [{"a": 1, "b": 1},{"a": 2},{"a": 3},{"a": 4}]
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
    """Balances by a signature that reflects the total length of the fields' values, quantized into integer segments.

    Args:
        segments_boundaries (List[int]): distinct integers sorted in increasing order, that maps a given total length
        into the index of the least of them that exceeds the total length. (If none exceeds -- into one index
        beyond, namely, the length of segments_boudaries)

        fields (Optional, List[str])

    Example:
        when input [{"a": [1, 3], "b": 0, "id": 0}, {"a": [1, 3], "b": 0, "id": 1}, {"a": [], "b": "a", "id": 2}] is fed into

        .. code-block::

            LengthBalancer(fields=["a"], segments_boundaries=[1])

        input instances will be counted and balanced against two categories: empty total length (less than 1), and non-empty.
    """

    segments_boundaries: List[int]
    fields: Optional[List[str]]

    def signature(self, instance):
        total_len = 0
        for field_name in self.fields:
            total_len += len(dict_get(instance, field_name, use_dpath=True))
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

    Attributes:
        source (str): URL of the file to be downloaded.
        target (str): Local path where the downloaded file should be saved.
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

    Attributes:
        zip_file (str): Path of the zip file to be extracted.
        target_dir (str): Directory where the contents of the zip file will be extracted.
    """

    zip_file: str
    target_dir: str

    def process(self):
        with zipfile.ZipFile(self.zip_file) as zf:
            zf.extractall(self.target_dir)
