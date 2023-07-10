from .artifact import Artifact

from abc import ABC, abstractmethod
from typing import Dict


class Instruction(Artifact):
    @abstractmethod
    def __call__(self) -> str:
        pass


class TextualInstruction(Instruction):
    text: str

    def __call__(self) -> str:
        return self.text


from .collections import ListCollection


class InstructionsList(ListCollection):
    def verify(self):
        for instruction in self.items:
            assert isinstance(instruction, Instruction)


class InstructionsDict(Dict):
    def verify(self):
        for key, instruction in self.items():
            assert isinstance(instruction, Instruction)
