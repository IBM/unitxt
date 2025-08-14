import json
from abc import abstractmethod
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union

from .artifact import Artifact
from .collections import DictCollection, ListCollection
from .dataclass import NonPositionalField
from .dict_utils import dict_get, dict_set
from .error_utils import Documentation, UnitxtError
from .operator import InstanceOperator, Operator
from .random_utils import new_random_generator
from .serializers import (
    ConversationSerializer,
    DialogSerializer,
    ImageSerializer,
    ListSerializer,
    MultiTypeSerializer,
    NumberQuantizingSerializer,
    Serializer,
    SQLDatabaseAsSchemaSerializer,
    TableSerializer,
    ToolCallSerializer,
    ToolsSerializer,
    VideoSerializer,
)
from .settings_utils import get_constants
from .type_utils import isoftype, to_type_string

constants = get_constants()


class TemplateFormatKeyError(UnitxtError):
    def __init__(self, template, data, data_type, format_str, format_name):
        keys = ", ".join(data.keys())
        super().__init__(
            f"Available {data_type}s are [{keys}] "
            f"but {template.__class__.__name__}.{format_name} format requires a different ones: '{format_str}'",
            Documentation.ADDING_TEMPLATE,
        )


class Template(InstanceOperator):
    """The role of template is to take the fields of every instance and verbalize it.

    Meaning the template is taking the instance and generating source, target and references.

    Args:
        skip_rendered_instance (bool): if "source", "target", and "references" are already defined fields in the instance, skip its processing
        postprocessors: a list of strings being artifact names of text processors, to be applied on the model output
        instruction: a formatting string that yields an instruction with potential participation of values from the "input_fields" part of the instance
        target_prefix: a string to be used to format the prompt. Not a formatting string.

    """

    skip_rendered_instance: bool = NonPositionalField(default=True)
    postprocessors: List[str] = NonPositionalField(
        default_factory=lambda: ["processors.to_string_stripped"]
    )
    instruction: str = NonPositionalField(default="")
    target_prefix: str = NonPositionalField(default="")
    title_fields: List[str] = NonPositionalField(default_factory=list)
    serializer: Serializer = NonPositionalField(
        default_factory=lambda: MultiTypeSerializer(
            serializers=[
                ImageSerializer(),
                VideoSerializer(),
                TableSerializer(),
                ToolCallSerializer(),
                ToolsSerializer(),
                DialogSerializer(),
                ConversationSerializer(),
                ListSerializer(),
                SQLDatabaseAsSchemaSerializer(),
            ]
        )
    )

    def verify(self):
        super().verify()
        assert isoftype(
            self.postprocessors, List[Union[Operator, str]]
        ), f"The template post processors field '{self.postprocessors}' is not a list of processors. Instead it is of type '{to_type_string(type(self.postprocessors))}'."

    def input_fields_to_instruction_and_target_prefix(self, input_fields, instruction):
        instruction = self.apply_formatting(
            input_fields, "input field", instruction, "instruction"
        )
        target_prefix = self.apply_formatting(
            input_fields,
            "input field",
            self.target_prefix,
            "target_prefix",
        )
        return instruction, target_prefix

    def preprocess_input_and_reference_fields(
        self, input_fields: Dict[str, Any], reference_fields: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return input_fields, reference_fields

    def preprocess_input_fields(self, input_fields: Dict[str, Any]):
        return input_fields

    def preprocess_reference_fields(self, reference_fields: Dict[str, Any]):
        return reference_fields

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.skip_rendered_instance:
            if (
                "source" in instance
                and "target" in instance
                and "references" in instance
            ):
                return instance

        input_fields = instance.get("input_fields")
        reference_fields = instance.get("reference_fields")

        if stream_name != constants.inference_stream:
            input_fields, reference_fields = self.preprocess_input_and_reference_fields(
                input_fields, reference_fields
            )

        input_fields = self.preprocess_input_fields(input_fields)

        self.set_titles(input_fields)

        serialized_inputs = self.serialize(input_fields, instance)

        source = self.input_fields_to_source(serialized_inputs)
        instruction, target_prefix = self.input_fields_to_instruction_and_target_prefix(
            serialized_inputs,
            instance.get(constants.instruction_field, self.instruction),
        )

        result = {
            **instance,
            "source": source,
            constants.instruction_field: instruction,
            "target_prefix": target_prefix,
            "postprocessors": self.postprocessors,
        }

        if stream_name == constants.inference_stream:
            return self.post_process_instance(result)

        if reference_fields is None:
            raise ValueError("Should have reference_fields")

        reference_fields = self.preprocess_reference_fields(reference_fields)

        serialized_references = self.serialize(
            reference_fields, instance
        )  # Dict[str, str]

        target, references = self.reference_fields_to_target_and_references(
            serialized_references
        )

        result["target"] = target
        result["references"] = references

        return self.post_process_instance(result)

    def post_process_instance(self, instance):
        return instance

    def serialize(
        self, data: Dict[str, Any], instance: Dict[str, Any]
    ) -> Dict[str, str]:
        return {k: self.serializer.serialize(v, instance) for k, v in data.items()}

    @abstractmethod
    def input_fields_to_source(self, input_fields: Dict[str, object]) -> str:
        pass

    def set_titles(self, data):
        for field in self.title_fields:
            data[field] = data[field].title()

    @abstractmethod
    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> Tuple[str, List[str]]:
        pass

    def apply_formatting(
        self, data: Dict[str, Any], data_type: str, format_str: str, format_name: str
    ) -> str:
        try:
            if format_str is None:
                raise UnitxtError(
                    f"Required field '{format_name}' of class {self.__class__.__name__} not set in {self.__class__.__name__}",
                    Documentation.ADDING_TEMPLATE,
                )
            return format_str.format(**data)
        except KeyError as e:
            raise TemplateFormatKeyError(
                self, data, data_type, format_str, format_name
            ) from e


class ApplyTemplate(InstanceOperator):
    demos_field: Optional[str] = None

    @abstractmethod
    def get_template(self, instance: Dict[str, Any]) -> Template:
        pass

    def apply(
        self,
        template: Template,
        instance: Dict[str, Any],
        stream_name: Optional[str] = None,
    ):
        return template.process_instance(instance, stream_name)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        template = self.get_template(instance)

        if self.demos_field is not None:
            if self.demos_field not in instance:
                raise ValueError("Demos field is missing.")
            instance[self.demos_field] = [
                self.apply(template, demo_instance)
                for demo_instance in instance[self.demos_field]
            ]
        dict_set(instance, "recipe_metadata/template", template)
        return self.apply(template, instance, stream_name)


class ApplySingleTemplate(ApplyTemplate):
    template: Template

    def get_template(self, instance: Dict[str, Any]) -> Template:
        return self.template


class ApplyRandomTemplate(ApplyTemplate):
    templates: List[Template]

    def get_template(self, instance: Dict[str, Any]) -> Template:
        random_generator = new_random_generator(
            {**instance["input_fields"], **instance["reference_fields"]}
        )
        return random_generator.choice(self.templates)


class InputFormatTemplate(Template):
    input_format: str

    def input_fields_to_source(self, input_fields: Dict[str, object]) -> str:
        return self.apply_formatting(
            input_fields,
            "input field",
            self.input_format,
            "input_format",
        )


class OutputFormatTemplate(Template):
    output_format: str = None

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        target = self.apply_formatting(
            reference_fields,
            "reference field",
            self.output_format,
            "output_format",
        )
        references = [target]
        return target, references


class JsonOutputFormatTemplate(Template):
    output_fields: Dict[str, str]
    wrap_with_list_fields: List[str]

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        data = {}
        for field, target_field in self.output_fields.items():
            value = reference_fields[field]
            if field in self.wrap_with_list_fields:
                value = [value]
            data[target_field] = value
        target = json.dumps(data, ensure_ascii=False)
        references = [target]
        return target, references


class InputOutputTemplate(InputFormatTemplate, OutputFormatTemplate):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify the formatting strings with which to glue together the input and reference fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    pass


class JsonOutputTemplate(InputFormatTemplate, JsonOutputFormatTemplate):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify the formatting strings with which to glue together the input and reference fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    pass


class InputOutputTemplateWithCustomTarget(InputOutputTemplate):
    reference: str

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        target = self.apply_formatting(
            reference_fields,
            "reference field",
            self.output_format,
            "output_format",
        )
        reference = self.apply_formatting(
            reference_fields,
            "reference field",
            self.reference,
            "reference",
        )
        return target, [reference]


class PairwiseChoiceTemplate(InputOutputTemplate):
    """PairwiseChoiceTemplate.

    Requirements:
     The answer field value should be of type Literal["choice_a", "choice_b", "tie"]

    Args:
         choice_a_field (str):
            The field which contains choice_a value
         choice_b_field (str):
            The field which contains choice_b value
         answer_field (str):
            The field which contains the answer value.
            Should be of type Literal["choice_1", "choice_2", "tie"]
         choice_a_label (str):
            The label of choice A answer as it is verbalized in the template.
         choice_b_label (str):
            The label of choice B answer as it is verbalized in the template.
         choice_tie_label (str):
            The label of a tie answer as it should be verbalized in the template.
         shuffle (bool):
            whether to shuffle the choices or not. This is done to take into account position bias.

    shuffle: 50% of the time:
     1. The values of choice_a_field and choice_b_field will be swapped.
     2. If the values of answer_field is choice_a_label, set it to choice_b_label.
        Else if the values of answer_field is choice_b_label, set it to choice_a_label.
        Else if the value of answer_field is choice_tie_label, do nothing.

    """

    choice_a_field: str
    choice_b_field: str
    answer_field: str
    choice_a_label: str
    choice_b_label: str
    choice_tie_label: str
    shuffle: bool

    def verify(self):
        super().verify()

    def verbalize_answer_field(self, reference_fields: Dict[str, object]):
        answer = reference_fields[self.answer_field]
        assert answer in ["choice_a", "choice_b", "tie"]
        if answer == "choice_a":
            reference_fields[self.answer_field] = self.choice_a_label
        elif answer == "choice_b":
            reference_fields[self.answer_field] = self.choice_b_label
        else:
            reference_fields[self.answer_field] = self.choice_tie_label

        return reference_fields

    def shuffle_values(
        self, input_fields: Dict[str, object], reference_fields: Dict[str, object]
    ):
        if not self.shuffle:
            return input_fields, reference_fields
        outcome = random()  # A float between 0 and 1
        if outcome <= 0.5:
            choice_a_value = input_fields[self.choice_a_field]
            choice_b_value = input_fields[self.choice_b_field]

            input_fields[self.choice_a_field] = choice_b_value
            input_fields[self.choice_b_field] = choice_a_value

            answer = reference_fields[self.answer_field]
            assert answer in [
                self.choice_a_label,
                self.choice_b_label,
                self.choice_tie_label,
            ]
            if answer == self.choice_a_label:
                reference_fields[self.answer_field] = self.choice_b_label
            elif answer == self.choice_b_label:
                reference_fields[self.answer_field] = self.choice_a_label

        return input_fields, reference_fields

    def preprocess_input_and_reference_fields(
        self, input_fields: Dict[str, Any], reference_fields: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        reference_fields = self.verbalize_answer_field(reference_fields)
        input_fields, reference_fields = self.shuffle_values(
            input_fields, reference_fields
        )
        return input_fields, reference_fields


class DialogFieldsData(Artifact):
    user_role_label: str
    assistant_role_label: str
    system_role_label: str
    dialog_field: str


class DialogTemplate(InputOutputTemplate):
    dialog_fields: List[DialogFieldsData]
    turns_separator: str = "\n\n"
    label_separator: str = " "

    def process_dialog(self, input_fields: Dict[str, object]):
        for dialog_fields in self.dialog_fields:
            dialog = input_fields[dialog_fields.dialog_field]
            # TODO: update isoftype method to support Literal verification and check
            #  it's List[Tuple[Literal["user", "assistant", "system"], str]] (Issue #799)
            assert isoftype(dialog, List[Tuple[str, str]])

            user_role_label = dialog_fields.user_role_label
            assistant_role_label = dialog_fields.assistant_role_label
            system_role_label = dialog_fields.system_role_label

            dialog_str = ""
            for i, turn in enumerate(dialog):
                (turn_type, turn_text) = turn
                turns_separator = "" if i == 0 else self.turns_separator
                if turn_type == "user":
                    dialog_str += f"{turns_separator}{user_role_label}{self.label_separator}{turn_text}"
                elif turn_type == "assistant":
                    dialog_str += f"{turns_separator}{assistant_role_label}{self.label_separator}{turn_text}"
                elif turn_type == "system":
                    dialog_str += f"{turns_separator}{system_role_label}{self.label_separator}{turn_text}"

            input_fields[dialog_fields.dialog_field] = dialog_str
        return input_fields

    def preprocess_input_fields(self, input_fields: Dict[str, Any]):
        return self.process_dialog(input_fields)


class DialogPairwiseChoiceTemplate(DialogTemplate, PairwiseChoiceTemplate):
    pass


class PairwiseComparativeRatingTemplate(InputOutputTemplate):
    """PairwiseChoiceTemplate.

    Args:
         choice_a_field (str): The field which contains choice_a value

         choice_b_field (str): The field which contains choice_b value

         answer_field (str): The field which contains the answer value. The value should be an int.
         Positive for preferring choice_a, and negative for preferring choice_b

         shuffle (bool): whether to shuffle the choices or not. This is done to take into account position bias.

    shuffle: 50% of the time:
    | 1) The values of choice_a_field and choice_b_field will be swapped.
    | 2) Replace the values of answer_field with its mapped value according to the reverse_preference_map Dict.

    """

    choice_a_field: str
    choice_b_field: str
    choice_a_id_field: str
    choice_b_id_field: str
    answer_field: str
    shuffle: bool

    def shuffle_values(
        self, input_fields: Dict[str, object], reference_fields: Dict[str, object]
    ):
        if not self.shuffle:
            return input_fields, reference_fields
        outcome = random()  # A float between 0 and 1
        if outcome <= 0.5:
            choice_a_value = input_fields[self.choice_a_field]
            choice_b_value = input_fields[self.choice_b_field]
            input_fields[self.choice_a_field] = choice_b_value
            input_fields[self.choice_b_field] = choice_a_value

            choice_a_id_value = input_fields[self.choice_a_id_field]
            choice_b_id_value = input_fields[self.choice_b_id_field]
            input_fields[self.choice_a_id_field] = choice_b_id_value
            input_fields[self.choice_b_id_field] = choice_a_id_value

            assert isinstance(reference_fields[self.answer_field], int)
            reference_fields[self.answer_field] = (
                int(reference_fields[self.answer_field]) * -1
            )

        return input_fields, reference_fields

    def preprocess_input_and_reference_fields(
        self, input_fields: Dict[str, Any], reference_fields: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_fields, reference_fields = self.shuffle_values(
            input_fields, reference_fields
        )
        return input_fields, reference_fields


class MultipleChoiceTemplate(InputFormatTemplate):
    """Formats the input that specifies a multiple-choice question, with a list of possible answers to choose from, and identifies the correct answer.

    Args:
        target_prefix (str): Optional prefix that can be added before the target label in
            generated prompts or outputs.
        choices_field (str): The key under which the multiple choices are stored in the
            input and reference dictionaries.
        target_field (str): The key under which the correct choice is stored in the
            reference dictionary (can be integer index or textual label).
        choices_separator (str): A string used to join formatted
            choices (e.g. ", ").
        source_choice_format (str): A Python format string used for displaying each choice
            in the input fields (e.g. "{choice_numeral}. {choice_text}").
        target_choice_format (str): A Python format string used for displaying each choice
            in the target or final output (e.g. "{choice_numeral}").
        enumerator (str): Determines how choice numerals are enumerated. Possible values
            include "capitals", "lowercase", "numbers", or "roman".
        shuffle_choices (bool): If True, shuffle the choices. The shuffling seed can be
            set with `shuffle_choices_seed`.
        shuffle_choices_seed (int, optional): If provided, the choices are shuffled with
            this fixed integer seed for reproducibility.
        sort_choices_by_length (bool): If True, sorts choices
            by their length (ascending).
        sort_choices_alphabetically (bool): If True, sorts choices
            in alphabetical order.
        reverse_choices (bool): If True, reverses the order of the choices after any
            sorting has been applied. Defaults to False to preserve backward compatibility.
    """

    target_prefix: str = ""
    choices_field: str = "choices"
    target_field: str = "label"
    choices_separator: str = ", "
    source_choice_format: str = "{choice_numeral}. {choice_text}"
    target_choice_format: str = "{choice_numeral}"
    enumerator: str = "capitals"

    shuffle_choices: bool = False
    shuffle_choices_seed: int = None
    sort_choices_by_length: bool = False
    sort_choices_alphabetically: bool = False
    reverse_choices: bool = False  # False by default for backward-compat
    place_correct_choice_position: int = None

    def prepare(self):
        super().prepare()
        if self.enumerator == "capitals":
            self.enumerator = "ABCDEFGHIJKLMNOP"
        if self.enumerator == "lowercase":
            self.enumerator = "abcdefghijklmnop"
        if self.enumerator == "numbers":
            self.enumerator = [str(i + 1) for i in range(20)]
        if self.enumerator == "roman":
            self.enumerator = [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
                "XI",
                "XII",
                "XIII",
                "XIV",
                "XV",
                "XVI",
                "XVII",
                "XVIII",
                "XIX",
                "XX",
            ]

    def verify(self):
        super().verify()
        if self.shuffle_choices and (
            self.sort_choices_by_length
            or self.sort_choices_alphabetically
            or self.reverse_choices
            or self.place_correct_choice_position is not None
        ):
            raise UnitxtError(
                "You cannot combine shuffle_choices with sorting or reversing flags."
            )

        if self.sort_choices_by_length and self.sort_choices_alphabetically:
            raise UnitxtError(
                "You cannot combine both sort_choices_by_length and sort_choices_alphabetically simultaneously."
            )
        if self.place_correct_choice_position is not None and (
            self.sort_choices_by_length
            or self.sort_choices_alphabetically
            or self.reverse_choices
        ):
            raise UnitxtError(
                "You cannot combine place_correct_choice_position with sorting or reversing flags."
            )

    def inputs_to_choices(self, data: Dict[str, Any], choice_format: str) -> str:
        choices = data[self.choices_field]
        enumrated_choices = []
        for i, choice in enumerate(choices):
            enumrated_choices.append(
                choice_format.format(
                    choice_text=choice,
                    choice_numeral=self.enumerator[i],
                )
            )
        return enumrated_choices

    def inputs_to_numerals(self, input_fields: Dict[str, Any]) -> Tuple[str, str]:
        return self.inputs_to_choices(input_fields, "{choice_numeral}")

    def prepare_multiple_choice_inputs(
        self, input_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        choices = self.inputs_to_choices(input_fields, self.source_choice_format)
        return {
            "numerals": self.inputs_to_numerals(input_fields),
            **input_fields,
            self.choices_field: self.choices_separator.join(choices),
        }

    def preprocess_input_fields(self, input_fields: Dict[str, Any]) -> Dict[str, Any]:
        return self.prepare_multiple_choice_inputs(input_fields)

    def outputs_to_target_index(self, reference_fields: Dict[str, object]) -> int:
        target = reference_fields[self.target_field]

        if not isinstance(target, int):
            try:
                return reference_fields[self.choices_field].index(target)
            except ValueError as e:
                raise UnitxtError(
                    f"MultipleChoiceTemplate could not locate textual target '{target}' in choices list: {reference_fields[self.choices_field]}",
                    Documentation.ADDING_TEMPLATE,
                ) from e
        return target

    def preprocess_reference_fields(self, reference_fields: Dict[str, Any]):
        target = reference_fields[self.target_field]

        if not isinstance(target, int):
            try:
                target = reference_fields[self.choices_field].index(target)
            except ValueError as e:
                raise UnitxtError(
                    f"MultipleChoiceTemplate could not locate textual target '{target}' in choices list: {reference_fields[self.choices_field]}",
                    Documentation.ADDING_TEMPLATE,
                ) from e

        choices = self.inputs_to_choices(reference_fields, self.target_choice_format)

        try:
            target = choices[target]
        except IndexError as e:
            raise UnitxtError(
                f"MultipleChoiceTemplate cannot find index number {target} in choices: {choices}",
                Documentation.ADDING_TEMPLATE,
            ) from e

        return {self.target_field: target}

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        target = reference_fields[self.target_field]
        return target, [target]

    def preprocess_input_and_reference_fields(
        self, input_fields: Dict[str, Any], reference_fields: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if (
            not self.shuffle_choices
            and not self.sort_choices_by_length
            and not self.sort_choices_alphabetically
            and not self.reverse_choices
            and self.place_correct_choice_position is None
        ):
            return input_fields, reference_fields

        choices = input_fields[self.choices_field]
        target_index = self.outputs_to_target_index(reference_fields)
        original_label_choice = reference_fields[self.choices_field][target_index]

        if self.sort_choices_by_length:
            choices.sort(key=len)
        if self.sort_choices_alphabetically:
            choices.sort()
        if self.reverse_choices:
            choices.reverse()
        if self.shuffle_choices:
            random_generator = new_random_generator(
                self.shuffle_choices_seed
                if self.shuffle_choices_seed is not None
                else {**input_fields}
            )
            random_generator.shuffle(choices)
        if self.place_correct_choice_position is not None:
            fix_pos = self.place_correct_choice_position

            # Supporting negative indexes similar to Python lists
            # If fix_pos is negative, convert it to a valid positive index by adding len(choices).
            # For example, -1 becomes the last index, -2 becomes the one before last, etc.
            if fix_pos < 0:
                fix_pos += len(choices)
            self.place_correct_choice_position = fix_pos
            # Remove the original label choice from the list
            if not 0 <= self.place_correct_choice_position < len(choices):
                raise ValueError(
                    f"fix_correct_choice_position={self.place_correct_choice_position} out of range (0..{len(choices) - 1})."
                )
            choices.remove(original_label_choice)
            choices.insert(self.place_correct_choice_position, original_label_choice)

        # Update both input_fields and reference_fields once at the end
        input_fields[self.choices_field] = choices
        reference_fields[self.choices_field] = choices
        reference_fields[self.target_field] = choices.index(original_label_choice)

        return input_fields, reference_fields

    def post_process_instance(self, instance):
        instance["input_fields"]["options"] = self.inputs_to_choices(
            instance["input_fields"], self.target_choice_format
        )
        return instance


class YesNoTemplate(InputFormatTemplate):
    """A template for generating binary Yes/No questions asking whether an input text is of a specific class.

    Args:
        input_format:
            Defines the format of the question.
        class_field:
            Defines the field that contains the name of the class that this template
            asks of.
        label_field:
            Defines the field which contains the true label of the input text. If a gold label is equal to the
            value in class_name, then the correct output is self.yes_answer (by default, "Yes").
            Otherwise the correct output is self.no_answer (by default, "No").
        yes_answer:
            The output value for when the gold label equals self.class_name.
            Defaults to "Yes".
        no_answer:
            The output value for when the gold label differs from self.class_name.
            Defaults to "No".
    """

    input_format: str = None
    class_field: str = None
    label_field: str = None
    yes_answer: str = "Yes"
    no_answer: str = "No"

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        try:
            gold_class_names = reference_fields[self.label_field]
        except KeyError as e:
            raise UnitxtError(
                f"Available reference_fields are {list(reference_fields.keys())}, missing required label field: '{self.label_field}'."
            ) from e
        if not isinstance(gold_class_names, list):
            raise UnitxtError(
                f"Unexpected value for gold_class_names: '{gold_class_names}'. Expecting a list."
            )
        try:
            queried_class_name = reference_fields[self.class_field]
        except KeyError as e:
            raise UnitxtError(
                f"Available reference_fields are {list(reference_fields.keys())}, missing required class field: '{self.class_field}'."
            ) from e
        if not queried_class_name or not isinstance(queried_class_name, str):
            raise UnitxtError(
                f"Unexpected value for queried_class_names: '{queried_class_name}'. Expected a string."
            )
        if queried_class_name in gold_class_names:
            return self.yes_answer, [self.yes_answer]
        return self.no_answer, [self.no_answer]


class NullTemplate(Template):
    """Templates that returns empty prompt and no references."""

    postprocessors = []

    def input_fields_to_source(self, input_fields: Dict[str, object]) -> str:
        return ""

    def reference_fields_to_target_and_references(self, reference_fields):
        return "", []


class KeyValTemplate(Template):
    """Generate field 'source' from fields designated as input, and fields 'target' and 'references' from fields designated as output, of the processed instance.

    Args specify with what separators to glue together the input and output designated fields of the processed instance into one string ('source' and 'target'), and into a list of strings ('references').
    """

    pairs_separator: str = ", "
    key_val_separator: str = ": "
    use_keys_for_inputs: bool = True
    outputs_key_val_separator: str = ": "
    use_keys_for_outputs: bool = False

    def process_dict(
        self, data: Dict[str, object], key_val_sep, pairs_sep, use_keys
    ) -> str:
        pairs = []
        for key, val in data.items():
            key_val = [key, str(val)] if use_keys else [str(val)]
            pairs.append(key_val_sep.join(key_val))
        return pairs_sep.join(pairs)

    def input_fields_to_source(self, input_fields: Dict[str, object]) -> str:
        return self.process_dict(
            input_fields,
            key_val_sep=self.key_val_separator,
            pairs_sep=self.pairs_separator,
            use_keys=self.use_keys_for_inputs,
        )

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> str:
        target = self.process_dict(
            reference_fields,
            key_val_sep=self.key_val_separator,
            pairs_sep=self.pairs_separator,
            use_keys=self.use_keys_for_outputs,
        )
        return target, [target]


class OutputQuantizingTemplate(InputOutputTemplate):
    serializer: MultiTypeSerializer = NonPositionalField(
        default_factory=MultiTypeSerializer
    )
    quantum: Union[float, int] = 0.1

    def prepare(self):
        super().prepare()
        self.serializer.add_serializers(
            [NumberQuantizingSerializer(quantum=self.quantum)]
        )


class MultiLabelTemplate(InputOutputTemplate):
    labels_field: str = "labels"
    labels_separator: str = ", "
    postprocessors = ["processors.to_list_by_comma"]
    output_format: str = "{labels}"
    empty_label: str = "None"

    def preprocess_reference_fields(
        self, reference_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        labels = reference_fields[self.labels_field]
        if not isinstance(labels, list):
            raise UnitxtError(
                f"MultiLabelTemplate requires labels field '{self.labels_field}' to be a list. Got {self.labels_field}<{type(labels).__name__}>: {labels}",
                Documentation.ADDING_TEMPLATE,
            )
        if len(labels) == 0:
            labels = [self.empty_label]
        labels_str = self.labels_separator.join(labels)
        return {self.labels_field: labels_str}


class MultiReferenceTemplate(InputOutputTemplate):
    references_field: str = "references"
    random_reference: bool = False
    serializer: Serializer = NonPositionalField(default_factory=MultiTypeSerializer)

    def serialize(
        self, data: Dict[str, Any], instance: Dict[str, Any]
    ) -> Dict[str, str]:
        result = {}
        for k, v in data.items():
            if k == self.references_field:
                v = [self.serializer.serialize(item, instance) for item in v]
            else:
                v = self.serializer.serialize(v, instance)
            result[k] = v
        return result

    def reference_fields_to_target_and_references(
        self, reference_fields: Dict[str, object]
    ) -> Tuple[str, List[str]]:
        references = reference_fields[self.references_field]
        if not isoftype(references, List[str]):
            raise UnitxtError(
                f"MultiReferenceTemplate requires references field '{self.references_field}' to be List[str]. Got {self.references_field}<{type(references).__name__}>: {references}",
                Documentation.ADDING_TEMPLATE,
            )
        if len(references) == 0:
            return "", []

        if self.random_reference:
            random_generator = new_random_generator(reference_fields)
            target = random_generator.choice(references)
        else:
            target = references[0]

        return target, references


class MultiTurnTemplate(MultiReferenceTemplate):
    input_format = ""
    turns_field: str

    def post_process_instance(self, instance):
        turns = dict_get(instance["input_fields"], self.turns_field)
        instance["__turns__"] = turns
        return super().post_process_instance(instance)


def escape_chars(s, chars_to_escape):
    for char in chars_to_escape:
        s = s.replace(char, f"\\{char}")
    return s


class SpanLabelingBaseTemplate(MultiLabelTemplate):
    spans_starts_field: str = "spans_starts"
    spans_ends_field: str = "spans_ends"
    text_field: str = "text"
    labels_support: list = None

    def extract_span_label_pairs(self, reference_fields):
        spans_starts = reference_fields[self.spans_starts_field]
        spans_ends = reference_fields[self.spans_ends_field]
        text = reference_fields[self.text_field]
        labels = reference_fields[self.labels_field]

        spans = []
        for span_start, span_end, label in zip(spans_starts, spans_ends, labels):
            if self.labels_support is None or label in self.labels_support:
                spans.append((span_start, span_end, text[span_start:span_end], label))

        for span in sorted(spans):
            if self.labels_support is None or span[3] in self.labels_support:
                yield span[2], span[3]

    def preprocess_reference_fields(
        self, reference_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        span_labels_pairs = self.extract_span_label_pairs(reference_fields)
        targets = self.span_label_pairs_to_targets(span_labels_pairs)
        return super().preprocess_reference_fields({"labels": targets})

    @abstractmethod
    def span_label_pairs_to_targets(self, pairs):
        pass


class SpanLabelingTemplate(SpanLabelingBaseTemplate):
    span_label_format: str = "{span}: {label}"
    escape_characters: List[str] = [":", ","]
    postprocessors: List[str] = ["processors.to_span_label_pairs"]

    def span_label_pairs_to_targets(self, span_label_pairs):
        targets = []
        for span, label in span_label_pairs:
            if self.escape_characters is not None:
                span = escape_chars(span, self.escape_characters)
            target = self.span_label_format.format(span=span, label=label)
            targets.append(target)
        return targets


class SpanLabelingJsonTemplate(SpanLabelingBaseTemplate):
    postprocessors = [
        "processors.load_json",
        "processors.dict_of_lists_to_value_key_pairs",
    ]

    def span_label_pairs_to_targets(self, span_label_pairs):
        groups = {}
        for span, label in span_label_pairs:
            if label not in groups:
                groups[label] = []
            groups[label].append(span)
        if len(groups) > 0:
            targets = [json.dumps(groups, ensure_ascii=False)]
        else:
            targets = []
        return targets


class TemplatesList(ListCollection):
    def verify(self):
        for template in self.items:
            assert isinstance(template, Template)


class TemplatesDict(DictCollection):
    def verify(self):
        for template in self.items.values():
            assert isinstance(template, Template)
