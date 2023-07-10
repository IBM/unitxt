from .operator import StreamInstanceOperator

from typing import Dict, List, Any


class Tasker:
    pass


class FormTask(Tasker, StreamInstanceOperator):
    inputs: List[str]
    outputs: List[str]
    metrics: List[str]

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        inputs = {key: instance[key] for key in self.inputs}
        outputs = {key: instance[key] for key in self.outputs}
        return {
            "inputs": inputs,
            "outputs": outputs,
            "metrics": self.metrics,
        }
