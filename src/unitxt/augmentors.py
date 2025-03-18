from random import Random
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from .operators import FieldOperator
from .random_utils import new_random_generator
from .type_utils import isoftype, parse_type_string, to_type_string
from .types import Text


class Augmentor(FieldOperator):
    """A stream operator that augments the values of either the task input fields before rendering with the template,  or the input passed to the model after rendering of the template."""

    pass


class TaskInputsAugmentor(Augmentor):
    def set_fields(self, fields: List[str]):
        fields = ["input_fields/" + field for field in fields]
        self.field_to_field = {field: field for field in fields}


class TypeDependentAugmentor(TaskInputsAugmentor):
    augmented_type: object

    def process_instance_value(self, value: Any, instance: Dict[str, Any]):
        if not isoftype(value, self.augmented_type):
            return value
        return super().process_instance_value(value=value, instance=instance)

    @classmethod
    def process_data_after_load(cls, data):
        if "augmented_type" in data:
            data["augmented_type"] = parse_type_string(data["augmented_type"])
        return data

    def process_data_before_dump(self, data):
        if "augmented_type" in data:
            data["augmented_type"] = to_type_string(data["augmented_type"])
        return data


class TextAugmentor(TypeDependentAugmentor):
    augmented_type = Text


class NullAugmentor(TaskInputsAugmentor):
    """Does not change the input string."""

    def process_value(self, value: Any) -> Any:
        return value


class AugmentWhitespace(TextAugmentor):
    """Augments the inputs by replacing existing whitespaces with other whitespaces.

    Currently, each whitespace is replaced by a random choice of 1-3 whitespace characters (space, tab, newline).
    """

    def process_value(self, value: str) -> str:
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


class AugmentPrefixSuffix(TextAugmentor):
    r"""Augments the input by prepending and appending randomly selected patterns (typically, whitespace).

    Args:
       prefixes (list or dict or None): the potential patterns (typically, whitespace) to select prefix from. The dictionary version allows the specification of relative weights for the different patterns. Set to None if not needed (i.e., only suffixes are needed).

       suffixes (list or dict or None): the potential patterns (typically, whitespace) to select suffix from. The dictionary version allows the specification of relative weights for the different patterns. Set to None if not needed (i.e., only prefixes are needed).

       prefix_len (positive int): the length of the prefix to be added.

       suffix_len (positive int): The length of the suffix to be added.

       remove_existing_whitespaces (bool): Clean any existing leading and trailing whitespaces. The selected pattern(s) are then prepended and/or appended to the potentially trimmed input.

    Examples:
        To prepend the input with a prefix made of 4 ``\n``-s or ``\t``-s, employ
        ``AugmentPrefixSuffix(augment_model_input=True, prefixes=['\n','\t'], prefix_len=4, suffixes = None)``.

        To append the input with a suffix made of 3 ``\n``-s or ``\t``-s, with ``\n`` being preferred over ``\t``,
        at 2:1 ratio, employ
        ``AugmentPrefixSuffix(augment_model_input=True, suffixes={'\n':2,'\t':1}, suffix_len=3, prefixes = None)``
        which will append ``\n``-s twice as often as ``\t``-s.

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
