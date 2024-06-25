from typing import Any, Dict, Optional

from .api import infer
from .artifact import Artifact
from .inference import InferenceEngine
from .operator import InstanceOperator


class Function(Artifact):
    recipe: str
    engine: InferenceEngine

    def __call__(self, **instance):
        return infer(instance, self.recipe, self.engine)


class FunctionOperator(InstanceOperator):
    function: Function

    def process(
        self, instance: Dict[str, Any], stream_name: Optional[str] = None
    ) -> Dict[str, Any]:
        return self.function(instance)
