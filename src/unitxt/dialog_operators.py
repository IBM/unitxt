"""Dialog Serializers.

Dialog serializers are the way to take dialog data and turn it into
text that can be fed to the model.

The format of the dialog is:

dialog = [
    {"user": "kkk", "agent": ""},
    {"user": "kkk", "agent": ""},
]

"""
from typing import Generator, Optional

from unitxt.stream import Stream

from .operators import SingleStreamOperator


class DialogSerializer(SingleStreamOperator):
    field: str
    to_field: str = None
    last_user_turn_to_field: str = None
    last_system_turn_to_field: str = None
    yield_every_turn: bool = False
    context_field: str = None

    def serialize_turn(self, turn, turn_format, without_system=False):
        if without_system:
            turn_format = turn_format[: turn_format.index("{user}") + len("{user}")]
            turn = {"user": turn["user"]}

        return turn_format.format(**turn)

    def process_turn(self, dialog, last_turn, turn_format):
        to_field = self.to_field if self.to_field is not None else self.field
        new_dialog = dialog + self.serialize_turn(last_turn, turn_format)

        if (
            self.last_system_turn_to_field is not None
            and self.last_user_turn_to_field is not None
        ):
            result = {
                self.last_system_turn_to_field: last_turn["system"],
                self.last_user_turn_to_field: last_turn["user"],
                to_field: dialog,
            }
        elif self.last_system_turn_to_field is not None:
            result = {
                self.last_system_turn_to_field: last_turn["system"],
                to_field: dialog
                + self.serialize_turn(last_turn, turn_format, without_system=True),
            }
        elif self.last_user_turn_to_field is not None:
            raise ValueError(
                "if last_user_turn_to_field is not None then last_system_turn_to_field should be not None."
            )
        else:
            result = {to_field: new_dialog}

        return result, new_dialog

    def get_turn_format(self, instance):
        turn_fomrat = instance["recipe_metadata"]["format"].demo_format
        turn_fomrat = turn_fomrat.replace("{source}", "{user}")
        turn_fomrat = turn_fomrat.replace("{target}", "{system}")
        return turn_fomrat.replace("{target_prefix}", "")

    def process(self, stream: Stream, stream_name: Optional[str] = None) -> Generator:
        for instance in stream:
            structred_dialog = instance[self.field]
            general_turn_format = self.get_turn_format(instance)

            dialog = "" if self.context_field is None else instance[self.context_field]
            for i, turn in enumerate(structred_dialog):
                turn_format = general_turn_format
                if i == 0:
                    turn_format = turn_format[turn_format.index("{user}") :]
                if i == len(structred_dialog) - 1:
                    turn_format = turn_format[
                        : turn_format.index("{system}") + len("{system}")
                    ]
                if (
                    i == len(structred_dialog) - 2
                    and self.last_user_turn_to_field is not None
                    and self.last_system_turn_to_field is not None
                ):
                    turn_format = turn_format[
                        : turn_format.index("{system}") + len("{system}")
                    ]

                result, dialog = self.process_turn(dialog, turn, turn_format)

                if self.yield_every_turn:
                    yield {**instance, **result}

            if not self.yield_every_turn:
                yield {**instance, **result}
