from typing import Any, Dict, List

from .operator import StreamInstanceOperator


class Tasker:
    pass


class FormTask(Tasker, StreamInstanceOperator):
    inputs: List[str]
    outputs: List[str]
    metrics: List[str]

    def process(self, instance: Dict[str, Any], stream_name: str = None) -> Dict[str, Any]:
        try:
            inputs = {key: instance[key] for key in self.inputs}
        except KeyError as e:
            raise KeyError(
                f"Unexpected input column names: {list(key for key in self.inputs if key not in instance)}"
                f"\n available names:{list(instance.keys())}\n given input names:{self.inputs}")
        try:
            outputs = {key: instance[key] for key in self.outputs}
        except KeyError as e:
            raise KeyError(
                f"Unexpected output column names: {list(key for key in self.inputs if key not in instance)}"
                f" \n available names:{list(instance.keys())}\n given output names:{self.outputs}")

        return {
            "inputs": inputs,
            "outputs": outputs,
            "metrics": self.metrics,
        }
