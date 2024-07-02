"""Dialog Serializers.

Dialog serializers are the way to take dialog data and turn it into
text that can be fed to the model.

The format of the dialog is:

dialog = [
    {"user": "hello", "system": "hi"},
    {"user": "kkk", "system": ""},
    {"user": "kkk", "system": ""},
]
"""

from typing import Any, Dict, List, Optional

from .formats import SystemFormat
from .operators import InstanceFieldOperator


class SerializeDialog(InstanceFieldOperator):
    """Serializes dialog data for feeding into a model.

    This class takes structured dialog data and converts it into a text format
    according to a specified template. It allows for the inclusion or exclusion
    of system responses and can operate on a per-turn basis or aggregate the entire
    dialog.

    Attributes:
        field (str): The field in the input data that contains the dialog.
        to_field (Optional[str]): The field in the output data where the serialized dialog will be stored.
        last_user_turn_to_field (Optional[str]): Field to store the last user turn.
        last_system_turn_to_field (Optional[str]): Field to store the last system turn.
        context_field (Optional[str]): Field that contains additional context to be prepended to the dialog.
    """

    format: Optional[SystemFormat] = None
    last_response_to_field: Optional[str] = None
    context_field: Optional[str] = None
    context_separator: str = " "

    def standardize_format(self, demo_format):
        turn_format = demo_format.replace("{source}", "{user}")
        turn_format = turn_format.replace("{target}", "{system}")
        return turn_format.replace("{target_prefix}", "")

    def slice_first_turn(self, turn_format):
        return turn_format[turn_format.index("{user}") :]

    def slice_last_turn(self, turn_format):
        return turn_format[: turn_format.index("{system}") + len("{system}")]

    def slice_last_response(self, turn_format):
        return turn_format[: turn_format.index("{user}") + len("{user}")]

    def get_turn_format(self, turn_format, step, length):
        if step == 0:
            turn_format = self.slice_first_turn(turn_format)
        if step == length - 1:
            turn_format = self.slice_last_turn(turn_format)
            if self.last_response_to_field is not None:
                turn_format = self.slice_last_response(turn_format)
        return turn_format

    def get_general_turn_format(self, instance):
        general_format = (
            instance["recipe_metadata"]["format"]
            if self.format is None
            else self.format
        )
        return self.standardize_format(general_format.demo_format)

    def process_instance_value(
        self, structured_dialog: List[Dict[str, str]], instance: Dict[str, Any]
    ):
        dialog = (
            ""
            if self.context_field is None
            else instance[self.context_field] + self.context_separator
        )
        general_turn_format = self.get_general_turn_format(instance)
        for i, turn in enumerate(structured_dialog):
            turn_format = self.get_turn_format(
                general_turn_format, i, len(structured_dialog)
            )
            dialog += turn_format.format(**turn)
        if self.last_response_to_field is not None:
            instance[self.last_response_to_field] = turn["system"]
        return dialog
