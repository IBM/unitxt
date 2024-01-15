import json
from typing import Any, List, Optional, Union

import requests

from ..logging_utils import get_logger
from ..metrics import GlobalMetric, Metric
from ..stream import MultiStream
from ..type_utils import isoftype

logger = get_logger()


class RemoteMetric(Metric):
    endpoint: str
    metric_name: str
    api_key: str = None

    @property
    def main_score(self):
        return None

    def get_metric_url(self) -> str:
        return f"{self.endpoint}/{self.metric_name}"


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
    metric: Metric | RemoteMetric,
    predictions: List[Any],
    references: List[List[Any]],
    additional_inputs: Optional[List[dict]] = None,
):
    assert isoftype(metric, Union[Metric, RemoteMetric]), "metric must be a Metric"
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
    if isinstance(metric, RemoteMetric):
        response = requests.post(
            url=metric.get_metric_url(),
            json=test_iterable,
            headers={"Authorization": f"Bearer {metric.api_key}"},
        )
        response.raise_for_status()
        result = response.json()
    else:
        output_multi_stream = metric(multi_stream)
        output_stream = output_multi_stream["test"]
        result = list(output_stream)
    return result


def test_metric(
    metric: Metric | RemoteMetric,
    predictions: List[Any],
    references: List[List[Any]],
    instance_targets: List[dict],
    global_target: dict,
    additional_inputs: Optional[List[dict]] = None,
):
    assert isoftype(metric, Union[Metric, RemoteMetric]), "operator must be an Operator"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[Any]), "references must be a list"

    if isinstance(metric, GlobalMetric) and metric.n_resamples:
        metric.n_resamples = 3  # Use a low number of resamples in testing for GlobalMetric, to save runtime
    outputs = apply_metric(
        metric,
        predictions,
        references,
        additional_inputs,
    )

    errors = []
    global_score = round_floats(outputs[0]["score"]["global"])
    if not dict_equal(global_score, global_target):
        errors.append(
            f"global score must be equal, got {json.dumps(global_score, sort_keys=True, ensure_ascii=False)} =/= "
            f"{json.dumps(global_target, sort_keys=True, ensure_ascii=False)}"
        )

    if len(outputs) == len(instance_targets):
        for output, instance_target in zip(outputs, instance_targets):
            instance_score = round_floats(output["score"]["instance"])
            if not dict_equal(instance_score, instance_target):
                errors.append(
                    f"instance score must be equal, "
                    f"got {json.dumps(instance_score, sort_keys=True, ensure_ascii=False)} =/= "
                    f"{json.dumps(instance_target, sort_keys=True, ensure_ascii=False)}"
                )
    else:
        errors.append(
            f"Metric outputs count does not match instance targets count, got {len(outputs)} =/= "
            f"{len(instance_targets)}"
        )

    if len(errors) > 0:
        raise AssertionError("\n".join(errors))

    logger.info("Metric tested successfully!")
    return True
