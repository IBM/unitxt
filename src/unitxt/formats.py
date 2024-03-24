from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from .dataclass import OptionalField
from .operator import StreamInstanceOperator
from .type_utils import isoftype


class Format(StreamInstanceOperator):
    pass


class SystemFormat(Format):
    r"""Generates the whole input to the model, from constant strings that are given as args, and from values found in specified fields of the instance.

    SystemFormat expects the input instance to contain:
    1. A field named "system_prompt" whose value is a string (potentially empty) that delivers a task independent opening text.
    2. A field named "source" whose value is a string verbalizing the original values in the instance (as read
    from the source dataset), in the context of the underlying task.
    3. A field named "instruction" that contains a (non-None) string.
    4. A field named with the value in arg 'demos_field', containing a list of dicts, each dict with fields "source"
    and "target", representing a single demo.
    5. A field named "target_prefx" that contains a string to prefix the target in both each demo, and to end the whole generated prompt

    SystemFormat formats the above fields into a single string to be inputted to the model. This string overwrites
    field "source" of the instance. Formatting is driven by two args: 'demo_format' and 'model_input_format'.
    SystemFormat also pops fields "system_prompt", "instruction", "target_prefix",  and the field containing the demos out from the input instance.

    Args:
        demos_field (str): the name of the field that contains the demos, being a list of dicts, each with "source" and "target" keys
        demo_format (str): formatting string for a single demo, combining fields "source" and "target"
        model_input_format (str) overall product format, combining instruction and source (as read from fields "instruction"
        and "source" of the input instance), together with demos (as formatted into one string)
        format_args: Dict[str,str]: additional format args to be used when formatting the different format strings

    Example:
        when input instance:

        .. code-block::

            {
                "source": "1+1",
                "target": "2",
                "instruction": "Solve the math exercises.",
                "demos": [{"source": "1+2", "target": "3"}, {"source": "4-2", "target": "2"}]
            }

        is processed by

        .. code-block::

            system_format = SystemFormat(
                demos_field="demos",
                demo_format="Input: {source}\nOutput: {target}\n\n",
                model_input_format="Instruction: {instruction}\n\n{demos}Input: {source}\nOutput: ",
            )

        the resulting instance is:

        .. code-block::

            {
                "target": "2",
                "source": "Instruction: Solve the math exercises.\n\nInput: 1+2\nOutput: 3\n\nInput: 4-2\nOutput: 2\n\nInput: 1+1\nOutput: ",
            }

    """

    demos_field: str = "demos"
    demo_format: str = "{source}\n{target_prefix}{target}\n\n"  #  example: "User: {source}\nAgent: {target}\n\n"
    model_input_format: str = (
        "{system_prompt}{instruction}{demos}{source}\n{target_prefix}"
    )
    format_args: Dict[str, str] = OptionalField(default_factory=dict)

    @staticmethod
    def _retrieve_field_and_assert_not_none(instance, field_name) -> str:
        if field_name is not None and field_name in instance:
            field_value = instance[field_name]
            assert (
                field_value is not None
            ), f"Value in field '{field_name}' should not be none. Received instance: {instance}"
            return field_value
        return ""

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        assert (
            "source" in instance
        ), f"field 'source' is expected to be in the input instance. Received instance: {instance}"
        source = self._retrieve_field_and_assert_not_none(
            instance=instance, field_name="source"
        )

        instruction = self._retrieve_field_and_assert_not_none(
            instance=instance, field_name="instruction"
        )
        target_prefix = self._retrieve_field_and_assert_not_none(
            instance=instance, field_name="target_prefix"
        )
        system_prompt = self._retrieve_field_and_assert_not_none(
            instance=instance, field_name="system_prompt"
        )

        # pop "system_prompt", "instruction", and "target_prefix" from instance
        if "target_prefix" in instance:
            instance.pop("target_prefix")
        if "instruction" in instance:
            instance.pop("instruction")
        if "system_prompt" in instance:
            instance.pop("system_prompt")

        demo_instances = []
        if self.demos_field is not None and self.demos_field in instance:
            demos = instance[self.demos_field]
            assert (
                demos is not None and isoftype(demos, List[Dict[str, Any]])
            ), f"A list of dict-s is expected in field '{self.demos_field}'. Received instance: {instance}"
            demo_instances = demos
            # pop demos from instance
            instance.pop(self.demos_field)

        demos_string = ""
        for demo_instance in demo_instances:
            demo_str = self.demo_format.format(
                target_prefix=target_prefix,
                source=demo_instance["source"],
                target=demo_instance["target"],
                **self.format_args,
            )
            demos_string += demo_str

        output = self.model_input_format.format(
            system_prompt=system_prompt,
            instruction=instruction,
            demos=demos_string,
            source=source,
            target_prefix=target_prefix,
            **self.format_args,
        )
        instance["source"] = output
        return instance
