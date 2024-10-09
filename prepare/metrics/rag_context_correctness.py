from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Rename
from unitxt.test_utils.metrics import test_evaluate, test_metric

base = "metrics.rag.context_correctness"
default = "mrr"

for new_catalog_name, base_catalog_name, main_score in [
    ("mrr", "metrics.mrr", "mrr"),
    ("map", "metrics.map", "map"),
    ("retrieval_at_k", "metrics.retrieval_at_k", "match_at_1"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="context_ids", to_field="prediction"),
            Wrap(
                field="ground_truths_context_ids", inside="list", to_field="references"
            ),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)


def test_context_correctness():
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
    retrieval_at_k_instance_targets = [
        {
            "match_at_1": 1.0,
            "match_at_3": 1.0,
            "match_at_5": 1.0,
            "match_at_10": 1.0,
            "match_at_20": 1.0,
            "match_at_40": 1.0,
            "precision_at_1": 1.0,
            "precision_at_3": 0.67,
            "precision_at_5": 0.67,
            "precision_at_10": 0.67,
            "precision_at_20": 0.67,
            "precision_at_40": 0.67,
            "recall_at_1": 0.5,
            "recall_at_3": 1.0,
            "recall_at_5": 1.0,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "recall_at_40": 1.0,
            "score": 1.0,
            "score_name": "match_at_1",
        },
        {
            "match_at_1": 0.0,
            "match_at_10": 1.0,
            "match_at_20": 1.0,
            "match_at_3": 1.0,
            "match_at_40": 1.0,
            "match_at_5": 1.0,
            "precision_at_1": 0.0,
            "precision_at_10": 0.5,
            "precision_at_20": 0.5,
            "precision_at_3": 0.5,
            "precision_at_40": 0.5,
            "precision_at_5": 0.5,
            "recall_at_1": 0.0,
            "recall_at_10": 1.0,
            "recall_at_20": 1.0,
            "recall_at_3": 1.0,
            "recall_at_40": 1.0,
            "recall_at_5": 1.0,
            "score": 0.0,
            "score_name": "match_at_1",
        },
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
    retrieval_at_k_global_target = {
        "match_at_1": 0.5,
        "match_at_1_ci_high": 1.0,
        "match_at_1_ci_low": 0.0,
        "match_at_3": 1.0,
        "match_at_5": 1.0,
        "match_at_10": 1.0,
        "match_at_20": 1.0,
        "match_at_40": 1.0,
        "precision_at_1": 0.5,
        "precision_at_1_ci_high": 1.0,
        "precision_at_1_ci_low": 0.0,
        "precision_at_3": 0.58,
        "precision_at_3_ci_high": 0.67,
        "precision_at_3_ci_low": 0.5,
        "precision_at_5": 0.58,
        "precision_at_5_ci_high": 0.67,
        "precision_at_5_ci_low": 0.5,
        "precision_at_10": 0.58,
        "precision_at_10_ci_high": 0.67,
        "precision_at_10_ci_low": 0.5,
        "precision_at_20": 0.58,
        "precision_at_20_ci_high": 0.67,
        "precision_at_20_ci_low": 0.5,
        "precision_at_40": 0.58,
        "precision_at_40_ci_high": 0.67,
        "precision_at_40_ci_low": 0.5,
        "recall_at_1": 0.25,
        "recall_at_1_ci_high": 0.5,
        "recall_at_1_ci_low": 0.0,
        "recall_at_3": 1.0,
        "recall_at_5": 1.0,
        "recall_at_10": 1.0,
        "recall_at_20": 1.0,
        "recall_at_40": 1.0,
        "score": 0.5,
        "score_ci_high": 1.0,
        "score_ci_low": 0.0,
        "score_name": "match_at_1",
    }

    for catalog_name, global_target, instance_targets, main_score in [
        (
            "metrics.rag.context_correctness.map",
            map_global_target,
            map_instance_targets,
            "map",
        ),
        (
            "metrics.rag.context_correctness.mrr",
            mrr_global_target,
            mrr_instance_targets,
            "mrr",
        ),
        (
            "metrics.rag.context_correctness",
            mrr_global_target,
            mrr_instance_targets,
            "mrr",
        ),
        (
            "metrics.rag.context_correctness.retrieval_at_k",
            retrieval_at_k_global_target,
            retrieval_at_k_instance_targets,
            "match_at_1",
        ),
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
            main_score=main_score,
            preprocess_steps=[
                Rename(field_to_field={"task_data/context_ids": "context_ids"}),
                Rename(
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


test_context_correctness()
