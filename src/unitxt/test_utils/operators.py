import json
from typing import List

from ..operator import StreamingOperator
from ..stream import MultiStream
from ..type_utils import isoftype
from .artifact import test_artfifact_saving_and_loading


def apply_operator(operator: StreamingOperator, inputs: List[dict], return_multi_stream=False, return_stream=False):
    if inputs is not None:
        multi_stream = MultiStream({"test": inputs})
        output_multi_stream = operator(multi_stream)
    else:
        output_multi_stream = operator()
    if return_multi_stream:
        return output_multi_stream
    output_stream = output_multi_stream["test"]
    if return_stream:
        return output_stream
    return list(output_stream)


def test_operator(
    operator: StreamingOperator, inputs: List[dict], targets: List[dict], tester=None, sort_outputs_by=None
):
    test_artfifact_saving_and_loading(operator, tester=tester)

    assert isoftype(operator, StreamingOperator), "operator must be an Operator"
    assert inputs is None or isoftype(inputs, List[dict]), "inputs must be a list of dicts or None for stream source"
    assert isoftype(targets, List[dict]), "outputs must be a list of dicts"

    outputs = apply_operator(operator, inputs)

    if sort_outputs_by is not None:
        outputs = sorted(outputs, key=lambda x: x[sort_outputs_by])

    if tester is None:
        errors = []

        for output, target in zip(outputs, targets):
            if json.dumps(output, sort_keys=True) == json.dumps(target, sort_keys=True):
                errors.append(f"input and output must be equal, got <{output}> =/= <{target}>")

        if len(errors) > 0:
            raise AssertionError("\n".join(errors))

        return True
    else:
        if inputs is None:
            inputs = [None] * len(targets)
        for input, output, target in zip(inputs, outputs, targets):
            with tester.subTest(operator=operator, input=input):
                tester.assertDictEqual(output, target)
