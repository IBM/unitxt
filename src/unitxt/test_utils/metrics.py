import json
from typing import Any, Dict, List

from ..metrics import Metric
from ..stream import MultiStream, Stream
from ..type_utils import isoftype


def round_floats(obj, precision=2, recursive=True):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict) and recursive:
        return {k: round_floats(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)) and recursive:
        return [round_floats(x) for x in obj]
    else:
        return obj


def dict_equal(dict1, dict2):
    return json.dumps(dict1, sort_keys=True) == json.dumps(dict2, sort_keys=True)


def apply_metric(
    metric: Metric,
    predictions: List[str],
    references: List[List[str]],
    inputs: List[dict] = None,
    outputs: List[dict] = None,
):
    assert isoftype(metric, Metric), "operator must be an Operator"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[Any]), "references must be a list"
    assert inputs is None or isoftype(inputs, List[Any]), "inputs must be a list"
    assert outputs is None or isoftype(outputs, List[Any]), "outputs must be a list"
    if inputs is not None and outputs is not None:
        test_iterable = [
            {"prediction": prediction, "references": reference, "inputs": inputs, "outputs": outputs}
            for prediction, reference, inputs, outputs in zip(predictions, references, inputs, outputs)
        ]
    else:
        test_iterable = [
            {"prediction": prediction, "references": reference}
            for prediction, reference in zip(predictions, references)
        ]
    multi_stream = MultiStream.from_iterables({"test": test_iterable}, copying=True)
    output_multi_stream = metric(multi_stream)
    output_stream = output_multi_stream["test"]
    return list(output_stream)


def test_metric(
    metric: Metric,
    predictions: List[str],
    references: List[List[str]],
    instance_targets: List[dict],
    global_target: dict,
    inputs: List[dict] = None,
    outputs: List[dict] = None,
):
    assert isoftype(metric, Metric), "operator must be an Operator"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[Any]), "references must be a list"

    outputs = apply_metric(metric, predictions, references, inputs, outputs)

    errors = []
    global_score = round_floats(outputs[0]["score"]["global"])
    if not dict_equal(global_score, global_target):
        errors.append(
            f"global score must be equal, got {json.dumps(global_score, sort_keys=True)} =/= {json.dumps(global_target, sort_keys=True)}"
        )

    for output, instance_target in zip(outputs, instance_targets):
        instance_score = round_floats(output["score"]["instance"])
        if not dict_equal(instance_score, instance_target):
            errors.append(
                f"instance score must be equal, got {json.dumps(instance_score, sort_keys=True)} =/= {json.dumps(instance_target, sort_keys=True)}"
            )

    if len(errors) > 0:
        raise AssertionError("\n".join(errors))

    print("Metric tested successfully!")
    return True
