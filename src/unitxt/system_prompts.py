from abc import abstractmethod
from typing import Any, Dict, Optional

from .dataclass import NonPositionalField
from .operator import InstanceOperator


class SystemPrompt(InstanceOperator):
    """The role of SystemPrompt is to add task-independent opening-text to every instance."""

    skip_rendered_instance: bool = NonPositionalField(default=True)

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.skip_rendered_instance:
            if "system_prompt" in instance:
                return instance

        instance["system_prompt"] = self.get_system_prompt(instance)

        return instance

    @abstractmethod
    def get_system_prompt(self, instance: Dict[str, object]) -> str:
        pass


class TextualSystemPrompt(SystemPrompt):
    """Specifies the system prompt as a totally independent string."""

    text: str

    def get_system_prompt(self, instance: Dict[str, object]) -> str:
        return self.text


class EmptySystemPrompt(SystemPrompt):
    def get_system_prompt(self, instance: Dict[str, object]) -> str:
        return ""
