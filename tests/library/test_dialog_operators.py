from unitxt.dialog_operators import SerializeDialog, SerializeOpenAiFormatDialog
from unitxt.test_utils.operators import check_operator

from tests.utils import UnitxtTestCase


class FormatMock:
    demo_format = "User: {source}\nAgent: {target_prefix}{target}\n"


class TestDialogSerializer(UnitxtTestCase):
    def test_dialog_serializer_with_last_system_turn_exclusion(self):
        self.maxDiff = None
        operator = SerializeDialog(
            field="dialog",
            to_field="serialized_dialog",
            last_response_to_field="last_system_turn",
            context_field=None,
        )

        inputs = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        targets = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "serialized_dialog": "Hello, how are you?\nAgent: I'm fine, thank you.\nUser: What's the weather like?",
                "last_system_turn": "It's sunny.",
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_dialog_serializer_with_no_exclusion(self):
        self.maxDiff = None
        operator = SerializeDialog(
            field="dialog",
            to_field="serialized_dialog",
            context_field=None,
        )

        inputs = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        targets = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "serialized_dialog": "Hello, how are you?\nAgent: I'm fine, thank you.\nUser: What's the weather like?\nAgent: It's sunny.",
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)

    def test_dialog_serializer_with_context(self):
        self.maxDiff = None
        operator = SerializeDialog(
            field="dialog",
            to_field="serialized_dialog",
            context_field="context",
        )

        inputs = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "context": "this is a context.",
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        targets = [
            {
                "dialog": [
                    {"user": "Hello, how are you?", "system": "I'm fine, thank you."},
                    {"user": "What's the weather like?", "system": "It's sunny."},
                ],
                "context": "this is a context.",
                "serialized_dialog": "this is a context. Hello, how are you?\nAgent: I'm fine, thank you.\nUser: What's the weather like?\nAgent: It's sunny.",
                "recipe_metadata": {"format": FormatMock},
            }
        ]

        check_operator(operator=operator, inputs=inputs, targets=targets, tester=self)


class TestOpenAiFormatDialogSerialize(UnitxtTestCase):
    dialog_operator = SerializeOpenAiFormatDialog(field="dummy_field")

    def test_transform_dialog(self):
        dialog = [
            {"content": "What color was Cotton?", "role": "user"},
            {"content": "white", "role": "assistant"},
            {"content": "Where did she live?", "role": "user"},
            {"content": "in a barn", "role": "assistant"},
            {"content": "Did she live alone?", "role": "user"},
            {"content": "No, with other animals", "role": "assistant"},
        ]

        expected = [
            {"user": "What color was Cotton?", "system": "white"},
            {"user": "Where did she live?", "system": "in a barn"},
            {"user": "Did she live alone?", "system": "No, with other animals"},
        ]

        assert (
            self.dialog_operator.transform_dialog_to_standard_format(dialog) == expected
        )

    def test_transform_dialog_with_turn_merging(self):
        # Test 'user', 'assistant', 'assistant'
        dialog = [
            {"content": "User", "role": "user"},
            {"content": "assistant1", "role": "assistant"},
            {"content": "assistant2", "role": "assistant"},
        ]

        expected = [{"user": "User", "system": "assistant1 assistant2"}]

        assert (
            self.dialog_operator.transform_dialog_to_standard_format(dialog) == expected
        )

        # Test multiple 'user' entries
        dialog = [
            {"content": "User1", "role": "user"},
            {"content": "User2", "role": "user"},
            {"content": "assistant1", "role": "assistant"},
            {"content": "assistant2", "role": "assistant"},
            {"content": "User3", "role": "user"},
            {"content": "assistant3", "role": "assistant"},
        ]

        expected = [
            {"user": "User1 User2", "system": "assistant1 assistant2"},
            {"user": "User3", "system": "assistant3"},
        ]

        assert (
            self.dialog_operator.transform_dialog_to_standard_format(dialog) == expected
        )

    def test_transform_dialog_validation_first_turn_assistant(self):
        # Test starting with 'system'
        dialog = [
            {"content": "assistant1", "role": "assistant"},
            {"content": "assistant2", "role": "assistant"},
            {"content": "user1", "role": "user"},
            {"content": "assistant3", "role": "assistant"},
        ]

        try:
            self.dialog_operator.transform_dialog_to_standard_format(dialog)
        except ValueError:
            return
        else:
            raise AssertionError(
                "Test failed: Exception not raised for invalid first role"
            )

    def test_transform_dialog_validation_invalid_role(self):
        dialog = [
            {"content": "User1", "role": "user"},
            {"content": "system1", "role": "system"},
        ]

        try:
            self.dialog_operator.transform_dialog_to_standard_format(dialog)
        except ValueError:
            return
        else:
            raise AssertionError("Test failed: Exception not raised for invalid role")
