import json
from typing import Any, List, Optional

from ..eval_utils import evaluate
from ..logging_utils import get_logger
from ..metrics import GlobalMetric, Metric
from ..settings_utils import get_settings
from ..stream import MultiStream
from ..type_utils import isoftype

logger = get_logger()
settings = get_settings()


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
    task_data: Optional[List[dict]] = None,
    perform_validations_in_apply_metric=True,
):
    if perform_validations_in_apply_metric:
        assert isoftype(metric, Metric), "metric must be a Metric"
        assert isoftype(predictions, List[Any]), "predictions must be a list"
        assert isoftype(
            references, List[List[Any]]
        ), "references must be a list of lists"
        assert len(references) == len(
            predictions
        ), "number of references and predictions elements must be equal"
        assert task_data is None or (
            len(references) == len(task_data)
        ), "number of references and task data elements must be equal"

        assert task_data is None or isoftype(
            task_data, List[Any]
        ), "task_data must be a list"

    if task_data is not None:
        test_iterable = [
            {
                "prediction": prediction,
                "references": reference,
                "task_data": additional_input,
            }
            for prediction, reference, additional_input in zip(
                predictions, references, task_data
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
    task_data: Optional[List[dict]] = None,
):
    if settings.test_metric_disable:
        logger.info(
            "test_metric() functionality is disabled because unitxt.settings.test_metric_disable=True or UNITXT_TEST_METRIC_DISABLE environment variable is set"
        )
        return None

    assert isoftype(metric, Metric), "operator must be an Operator"
    assert isoftype(predictions, List[Any]), "predictions must be a list"
    assert isoftype(references, List[Any]), "references must be a list"

    if isinstance(metric, GlobalMetric) and metric.n_resamples:
        metric.n_resamples = 3  # Use a low number of resamples in testing for GlobalMetric, to save runtime
    outputs = apply_metric(metric, predictions, references, task_data)

    check_scores(
        global_target,
        instance_targets,
        global_outputs=outputs[0]["score"]["global"],
        instance_outputs=[output["score"]["instance"] for output in outputs],
    )

    logger.info("Metric tested successfully!")
    return True


def check_scores(
    global_target: dict,
    instance_targets: List[dict],
    global_outputs: dict,
    instance_outputs: List[dict],
):
    errors = []
    global_score = round_floats(global_outputs)
    if not dict_equal(global_score, global_target):
        errors.append(
            f"global score must be equal, got {json.dumps(global_score, sort_keys=True, ensure_ascii=False)} =/= "
            f"{json.dumps(global_target, sort_keys=True, ensure_ascii=False)}"
        )
    if len(instance_outputs) == len(instance_targets):
        for i, instance_output, instance_target in zip(
            range(0, len(instance_outputs)), instance_outputs, instance_targets
        ):
            instance_score = round_floats(instance_output)
            if not dict_equal(instance_score, instance_target):
                errors.append(
                    f"instance {i} score must be equal, "
                    f"got {json.dumps(instance_score, sort_keys=True, ensure_ascii=False)} =/= "
                    f"{json.dumps(instance_target, sort_keys=True, ensure_ascii=False)}"
                )
    else:
        errors.append(
            f"Metric outputs count does not match instance targets count, got {len(instance_outputs)} =/= "
            f"{len(instance_targets)}"
        )
    if len(errors) > 0:
        raise AssertionError("\n".join(errors))


def test_evaluate(
    global_target: dict,
    instance_targets: List[dict],
    task_data: Optional[List[dict]],
    metric_name: str,
):
    evaluation_result, global_outputs = evaluate(
        task_data, metric_names=[metric_name], compute_conf_intervals=True
    )
    instance_outputs = [
        {
            "score": result[metric_name],
        }
        for result in evaluation_result
    ]
    check_scores(
        global_target, instance_targets, global_outputs[metric_name], instance_outputs
    )
