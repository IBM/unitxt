from src.unitxt.dialog_operators import SerializeDialog
from src.unitxt.test_utils.operators import check_operator
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
