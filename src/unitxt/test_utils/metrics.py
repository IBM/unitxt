import json
import os
from typing import Any, List, Optional

from ..logging_utils import get_logger
from ..metrics import GlobalMetric, Metric
from ..stream import MultiStream
from ..type_utils import isoftype

logger = get_logger()


def round_floats(obj, precision=2, recursive=True):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict) and recursive:
        return {k: round_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and recursive:
        return [round_floats(x) for x in obj]
    return obj


def dict_equal(dict1, dict2):
    return json.dumps(dict1, sort_keys=True, ensure_ascii=False) == json.dumps(
        dict2, sort_keys=True, ensure_ascii=False
    )


def apply_metric(
    metric: Metric,
    predictions: List[Any],
    references: List[List[Any]],
    additional_inputs: Optional[List[dict]] = None,
):
    assert isoftype(metric, Metric), "metric must be a Metric"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[List[Any]]), "references must be a list of lists"
    assert additional_inputs is None or isoftype(
        additional_inputs, List[Any]
    ), "additional_inputs must be a list"
    if additional_inputs is not None:
        test_iterable = [
            {
                "prediction": prediction,
                "references": reference,
                "additional_inputs": additional_input,
            }
            for prediction, reference, additional_input in zip(
                predictions, references, additional_inputs
            )
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
    predictions: List[Any],
    references: List[List[Any]],
    instance_targets: List[dict],
    global_target: dict,
    additional_inputs: Optional[List[dict]] = None,
):
    disable = os.getenv("UNITXT_TEST_METRIC_DISABLE", None)
    if disable is not None:
        logger.info(
            "test_metric() functionality is disabled because UNITXT_TEST_METRIC_DISABLE environment variable is set"
        )
        return None

    assert isoftype(metric, Metric), "operator must be an Operator"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[Any]), "references must be a list"

    if isinstance(metric, GlobalMetric) and metric.n_resamples:
        metric.n_resamples = 3  # Use a low number of resamples in testing for GlobalMetric, to save runtime
    outputs = apply_metric(metric, predictions, references, additional_inputs)

    errors = []
    global_score = round_floats(outputs[0]["score"]["global"])
    if not dict_equal(global_score, global_target):
        errors.append(
            f"global score must be equal, got {json.dumps(global_score, sort_keys=True, ensure_ascii=False)} =/= {json.dumps(global_target, sort_keys=True, ensure_ascii=False)}"
        )

    if len(outputs) == len(instance_targets):
        for output, instance_target in zip(outputs, instance_targets):
            instance_score = round_floats(output["score"]["instance"])
            if not dict_equal(instance_score, instance_target):
                errors.append(
                    f"instance score must be equal, got {json.dumps(instance_score, sort_keys=True, ensure_ascii=False)} =/= {json.dumps(instance_target, sort_keys=True, ensure_ascii=False)}"
                )
    else:
        errors.append(
            f"Metric outputs count does not match instance targets count, got {len(outputs)} =/= {len(instance_targets)}"
        )

    if len(errors) > 0:
        raise AssertionError("\n".join(errors))

    logger.info("Metric tested successfully!")
    return True
