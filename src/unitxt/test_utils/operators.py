import json
from typing import List

from ..operator import StreamingOperator
from ..stream import MultiStream
from ..type_utils import isoftype


def apply_operator(operator: StreamingOperator, inputs: List[dict], return_multi_stream=False, return_stream=False):
    multi_stream = MultiStream({"test": inputs})
    output_multi_stream = operator(multi_stream)
    if return_multi_stream:
        return output_multi_stream
    output_stream = output_multi_stream["test"]
    if return_stream:
        return output_stream
    return list(output_stream)


def test_operator(operator: StreamingOperator, inputs: List[dict], targets: List[dict]):
    assert isoftype(operator, StreamingOperator), "operator must be an Operator"
    assert isoftype(inputs, List[dict]), "inputs must be a list of dicts"
    assert isoftype(outputs, List[dict]), "outputs must be a list of dicts"

    outputs = apply_operator(operator, inputs)

    errors = []

    for input, output in zip(outputs, targets):
        if json.dumps(input, sort_keys=True) == json.dumps(output, sort_keys=True):
            errors.append(f"input and output must be equal, got <{input}> =/= <{output}>")

    if len(errors) > 0:
        raise AssertionError("\n".join(errors))

    return True
