"""Dialog Serializers.

Dialog serializers are the way to take dialog data and turn it into
text that can be fed to the model.

The format of the dialog is:

.. code-block:: text

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

    Args:
        field (str):
            The field in the input data that contains the dialog.
        to_field (Optional[str]):
            The field in the output data where the serialized dialog will be stored.
        last_user_turn_to_field (Optional[str]):
            Field to store the last user turn.
        last_system_turn_to_field (Optional[str]):
            Field to store the last system turn.
        context_field (Optional[str]):
            Field that contains additional context to be prepended to the dialog.
    """

    format: SystemFormat = None
    last_response_to_field: Optional[str] = None
    context_field: Optional[str] = None
    context_separator: str = " "
    slice_first_and_last_turns_format: bool = True

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
        if step == 0 and self.slice_first_and_last_turns_format:
            turn_format = self.slice_first_turn(turn_format)
        if step == length - 1:
            if self.slice_first_and_last_turns_format:
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


class SerializeOpenAiFormatDialog(SerializeDialog):
    """Serializes dialog data for feeding into a model.

    This class takes structured dialog data in the OpenAi format, and converts it into a text format
    according to a specified template. It allows for the inclusion or exclusion
    of system responses and can operate on a per-turn basis or aggregate the entire
    dialog.

    Args:
        field (str):
            The field in the input data that contains the dialog.
        to_field (Optional[str]):
            The field in the output data where the serialized dialog will be stored.
        last_user_turn_to_field (Optional[str]):
            Field to store the last user turn.
        last_system_turn_to_field (Optional[str]):
            Field to store the last system turn.
        context_field (Optional[str]):
            Field that contains additional context to be prepended to the dialog.
    """

    is_last_turn_user_only: bool = True

    @staticmethod
    def validate_openai_dialog_format(dialog: List[Dict[str, str]]) -> None:
        """Validates that the given dialog follows the correct OpenAI format.

        The function checks that:
        1. The dialog is a list of dictionaries.
        2. Each dictionary contains the keys 'role' and 'content'.
        3. The 'role' value is either 'user' or 'assistant'.
        4. Both 'role' and 'content' values are strings.
        5. The first 'role' is 'user'

        If the dialog does not conform to the expected format, a descriptive
        ValueError is raised indicating the issue.

        Args:
            dialog (List[Dict[str, str]]): The dialog to validate.

        Raises:
            ValueError: If the dialog does not meet the format requirements.
        """
        if not isinstance(dialog, list):
            raise ValueError("Dialog must be a list of dictionaries.")

        for i, entry in enumerate(dialog):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"Entry {i} is not a dictionary: {entry}. Each entry in the dialog must be a dictionary."
                )

            if "role" not in entry:
                raise ValueError(
                    f"Entry {i} is missing the 'role' key: {entry}. Each dictionary must have a 'role' key."
                )

            if "content" not in entry:
                raise ValueError(
                    f"Entry {i} is missing the 'content' key: {entry}. Each dictionary must have a 'content' key."
                )

            if not isinstance(entry["role"], str):
                raise ValueError(
                    f"Entry {i} has a non-string 'role': {entry['role']}. The 'role' value must be a string."
                )

            if not isinstance(entry["content"], str):
                raise ValueError(
                    f"Entry {i} has a non-string 'content': {entry['content']}. The 'content' value must be a string."
                )

            if entry["role"].lower() not in {"user", "assistant"}:
                raise ValueError(
                    f"Entry {i} has an invalid role: {entry['role']}. Allowed roles are 'user' and 'assistant'."
                )

        first_entry = dialog[0]
        if first_entry["role"].lower() != "user":
            raise ValueError(
                f"First entry role is expected to be 'user' It is  {first_entry['role']}."
            )

    @staticmethod
    def merge_dialog_entries(dialog: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Merges consecutive dialog entries with the same role.

        Args:
            dialog (List[Dict[str, str]]): The input dialog list where each dictionary has a 'role' and 'content'.

        Returns:
            List[Dict[str, str]]: A new list where consecutive entries with the same role are merged.
        """
        if len(dialog) == 0:
            return []

        merged_dialog = [dialog[0]]

        for entry in dialog[1:]:
            if entry["role"] == merged_dialog[-1]["role"]:
                merged_dialog[-1]["content"] += " " + entry["content"]
            else:
                merged_dialog.append(entry)

        return merged_dialog

    def transform_dialog_to_standard_format(
        self, dialog: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Transforms a dialog from OpenAI format to a simplified format.

        Each dictionary
        contains 'user' and 'system' keys with their respective contents. Consecutive entries
        with the same role are merged. Entries with invalid roles raise an error.

        Args:
            dialog (List[Dict[str, str]]): The input dialog in OpenAI format.

        Returns:
            List[Dict[str, str]]: The transformed dialog.

        Raises:
            ValueError: If an invalid role is detected.
        """
        SerializeOpenAiFormatDialog.validate_openai_dialog_format(dialog)
        merged_dialog = SerializeOpenAiFormatDialog.merge_dialog_entries(dialog)
        # self.validate_dialog_have_complete_pairs(merged_dialog)

        result = []
        for i in range(0, len(merged_dialog) - 1, 2):
            user_entry = merged_dialog[i]
            system_entry = merged_dialog[i + 1]

            result.append(
                {"user": user_entry["content"], "system": system_entry["content"]}
            )
        if len(merged_dialog) % 2 != 0:
            user_entry = merged_dialog[-1]
            result.append({"user": user_entry["content"], "system": ""})

        return result

    def process_instance_value(
        self, structured_dialog: List[Dict[str, str]], instance: Dict[str, Any]
    ):
        standard_format_dialog = self.transform_dialog_to_standard_format(
            structured_dialog
        )
        return super().process_instance_value(standard_format_dialog, instance)
