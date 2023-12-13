from abc import ABC
from typing import Any, Dict, List, Optional

from .dataclass import InternalField
from .formats import Format, ICLFormat
from .instructions import Instruction
from .operator import Operator, SequentialOperator, StreamInstanceOperator
from .templates import Template


class Renderer(ABC):
    pass
    # @abstractmethod
    # def get_postprocessors(self) -> List[str]:
    #     pass


class RenderDemonstrations(Renderer, StreamInstanceOperator):
    template: Template
    demos_field: str

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        demos = instance.get(self.demos_field, [])

        processed_demos = []
        for demo_instance in demos:
            demo_instance = self.template.process(demo_instance)
            processed_demos.append(demo_instance)

        instance[self.demos_field] = processed_demos

        return instance


class RenderInstruction(Renderer, StreamInstanceOperator):
    instruction: Instruction

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.instruction is not None:
            instance["instruction"] = self.instruction()
        else:
            instance["instruction"] = ""
        return instance


class RenderFormat(Renderer, StreamInstanceOperator):
    format: Format
    demos_field: str = None

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        demos_instances = instance.pop(self.demos_field, None)
        if demos_instances is not None:
            instance["source"] = self.format.format(
                instance, demos_instances=demos_instances
            )
        else:
            instance["source"] = self.format.format(instance)
        return instance


class StandardRenderer(Renderer, SequentialOperator):
    template: Template
    instruction: Instruction = None
    demos_field: str = None
    format: ICLFormat = None

    steps: List[Operator] = InternalField(default_factory=list)

    def prepare(self):
        self.steps = [
            self.template,
            RenderDemonstrations(template=self.template, demos_field=self.demos_field),
            RenderInstruction(instruction=self.instruction),
            RenderFormat(format=self.format, demos_field=self.demos_field),
        ]

    def get_postprocessors(self):
        return self.template.get_postprocessors()
