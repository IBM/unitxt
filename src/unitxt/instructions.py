from abc import abstractmethod
from typing import Dict

from .artifact import Artifact
from .collections import ListCollection


class Instruction(Artifact):
    @abstractmethod
    def __call__(self) -> str:
        pass


class TextualInstruction(Instruction):
    text: str

    def __call__(self) -> str:
        return self.text

    def __repr__(self):
        return self.text


class InstructionsList(ListCollection):
    def verify(self):
        for instruction in self.items:
            assert isinstance(instruction, Instruction)


class InstructionsDict(Dict):
    def verify(self):
        for _key, instruction in self.items():
            assert isinstance(instruction, Instruction)
