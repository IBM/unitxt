from abc import ABC
from typing import Any, Dict, Optional

from .operator import StreamInstanceOperator
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

    def get_postprocessors(self):
        return self.template.get_postprocessors()
