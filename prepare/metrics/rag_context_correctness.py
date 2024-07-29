from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import (
    Copy,
    RenameFields,
    RestorePredictionReferences,
    SavePredictionReferences,
)

for metric_name, catalog_name in [
    ("map", "metrics.rag.map"),
    ("mrr", "metrics.rag.mrr"),
    ("mrr", "metrics.rag.context_correctness"),
]:
    metric = MetricPipeline(
        main_score="score",
        preprocess_steps=[
            SavePredictionReferences(name=catalog_name),
            Copy(field="context_ids", to_field="prediction"),
            Wrap(
                field="ground_truths_context_ids", inside="list", to_field="references"
            ),
        ],
        metric=f"metrics.{metric_name}",
        postprocess_steps=[
            RestorePredictionReferences(name=catalog_name),
        ],
    )
    add_to_catalog(metric, catalog_name, overwrite=True)


if __name__ == "__main__":
    from unitxt.test_utils.metrics import test_evaluate, test_metric

    task_data = [
        {  # MRR is 1, MAP is (1 + 2/3)/2 = 0.833
            "context_ids": ["A", "B", "C"],
            "ground_truths_context_ids": ["A", "C"],
        },
        {  # MRR and MAP are both 0.5
            "context_ids": ["A", "B"],
            "ground_truths_context_ids": ["B"],
        },
    ]

    map_instance_targets = [
        {"map": 0.83, "score": 0.83, "score_name": "map"},
        {"map": 0.5, "score": 0.5, "score_name": "map"},
    ]
    mrr_instance_targets = [
        {"mrr": 1.0, "score": 1.0, "score_name": "mrr"},
        {"mrr": 0.5, "score": 0.5, "score_name": "mrr"},
    ]

    map_global_target = {
        "map": 0.67,
        "map_ci_high": 0.83,
        "map_ci_low": 0.5,
        "score": 0.67,
        "score_ci_high": 0.83,
        "score_ci_low": 0.5,
        "score_name": "map",
    }
    mrr_global_target = {
        "mrr": 0.75,
        "mrr_ci_high": 1.0,
        "mrr_ci_low": 0.5,
        "score": 0.75,
        "score_ci_high": 1.0,
        "score_ci_low": 0.5,
        "score_name": "mrr",
    }

    for catalog_name, global_target, instance_targets in [
        ("metrics.rag.map", map_global_target, map_instance_targets),
        ("metrics.rag.mrr", mrr_global_target, mrr_instance_targets),
        ("metrics.rag.context_correctness", mrr_global_target, mrr_instance_targets),
    ]:
        # test the evaluate call
        test_evaluate(
            global_target,
            instance_targets=[
                {"score": instance["score"]} for instance in instance_targets
            ],
            task_data=task_data,
            metric_name=catalog_name,
        )

        # test using the usual metric pipeline
        test_pipeline = MetricPipeline(
            main_score="score",
            preprocess_steps=[
                RenameFields(field_to_field={"task_data/context_ids": "context_ids"}),
                RenameFields(
                    field_to_field={
                        "task_data/ground_truths_context_ids": "ground_truths_context_ids"
                    }
                ),
            ],
            metric=f"{catalog_name}",
        )
        test_metric(
            metric=test_pipeline,
            predictions=[None, None],
            references=[[], []],
            instance_targets=instance_targets,
            global_target=global_target,
            task_data=task_data,
        )
