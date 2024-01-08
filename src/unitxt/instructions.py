from abc import abstractmethod
from typing import Any, Dict, Optional

from .collections import ListCollection
from .dataclass import NonPositionalField
from .operator import StreamInstanceOperator


class Instruction(StreamInstanceOperator):
    """The role of instruction is to add instruction to every instance.

    Meaning the instruction is taking the instance and generating instruction field for it.
    """

    skip_rendered_instance: bool = NonPositionalField(default=True)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.skip_rendered_instance:
            if "instruction" in instance:
                return instance

        instance["instruction"] = self.get_instruction(instance)

        return instance

    @abstractmethod
    def get_instruction(self, instance: Dict[str, object]) -> str:
        pass


class TextualInstruction(Instruction):
    text: str

    def get_instruction(self, instance: Dict[str, object]) -> str:
        return self.text


class EmptyInstruction(Instruction):
    def get_instruction(self, instance: Dict[str, object]) -> str:
        return ""


class InstructionsList(ListCollection):
    def verify(self):
        for instruction in self.items:
            assert isinstance(instruction, Instruction)


class InstructionsDict(Dict):
    def verify(self):
        for _key, instruction in self.items():
            assert isinstance(instruction, Instruction)
