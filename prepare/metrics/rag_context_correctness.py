from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Rename
from unitxt.test_utils.metrics import test_metric

default = "mrr"
base = "metrics.rag"
tasks = ["autorag", "end_to_end"]
dimension = "context_correctness"


def get_scores_prefix(metric_catalog_name, dim_name):
    return f"{dim_name}_"


def add_scores_prefix_to_target(target, metric_catalog_name, dim_name):
    prefix = get_scores_prefix(metric_catalog_name, dim_name)
    new_target = {
        f"{prefix}" + k
        if k not in ["score", "score_name", "num_of_instances"]
        and not k.startswith("score")
        else k: v
        for k, v in target.items()
    }
    new_target["score_name"] = prefix + new_target["score_name"]
    return new_target


def get_preprocess_steps(task):
    if task == "autorag":
        return [
            Copy(field="context_ids", to_field="prediction"),
            Wrap(
                field="ground_truths_context_ids", inside="list", to_field="references"
            ),
        ]
    if task == "end_to_end":
        return [
            Copy(field="prediction/context_ids", to_field="prediction"),
            Wrap(
                field="task_data/reference_context_ids",
                inside="list",
                to_field="references",
            ),
        ]
    raise ValueError(f"Unsupported rag task {task}")


def get_test_pipeline_task_preprocess_steps(task):
    if task == "autorag":
        return [
            Rename(field_to_field={"task_data/context_ids": "context_ids"}),
            Rename(
                field_to_field={
                    "task_data/ground_truths_context_ids": "ground_truths_context_ids"
                }
            ),
        ]
    if task == "end_to_end":
        return [
            Rename(field_to_field={"task_data/context_ids": "prediction/context_ids"}),
            Rename(
                field_to_field={
                    "task_data/ground_truths_context_ids": "task_data/reference_context_ids"
                }
            ),
        ]
    raise ValueError(f"Unsupported rag task for {dimension}:{task}")


for new_catalog_name, base_catalog_name, main_score in [
    ("mrr", "metrics.mrr", "mrr"),
    ("map", "metrics.map", "map"),
    ("retrieval_at_k", "metrics.retrieval_at_k", "match_at_1"),
]:
    for task in tasks:
        metric = MetricPipeline(
            main_score=main_score,
            preprocess_steps=get_preprocess_steps(task).copy(),
            metric=base_catalog_name,
            score_prefix=get_scores_prefix(new_catalog_name, dimension),
        )
        add_to_catalog(
            metric, f"{base}.{task}.{dimension}.{new_catalog_name}", overwrite=True
        )

        if new_catalog_name == default and task == "autorag":
            add_to_catalog(metric, f"{base}.{task}.{dimension}", overwrite=True)


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
        "num_of_instances": 2,
    }
    mrr_global_target = {
        "mrr": 0.75,
        "mrr_ci_high": 1.0,
        "mrr_ci_low": 0.5,
        "score": 0.75,
        "score_ci_high": 1.0,
        "score_ci_low": 0.5,
        "score_name": "mrr",
        "num_of_instances": 2,
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
        "num_of_instances": 2,
    }

    for catalog_name, global_target, instance_targets, main_score in [
        (
            f"{base}.{task}.{dimension}.map",
            map_global_target,
            map_instance_targets,
            "map",
        ),
        (
            f"{base}.{task}.{dimension}.mrr",
            mrr_global_target,
            mrr_instance_targets,
            "mrr",
        ),
        # (
        #     f"{base}.{task}.{dimension}",
        #     mrr_global_target,
        #     mrr_instance_targets,
        #     "mrr",
        # ),
        (
            f"{base}.{task}.{dimension}.retrieval_at_k",
            retrieval_at_k_global_target,
            retrieval_at_k_instance_targets,
            "match_at_1",
        ),
    ]:
        # # test the evaluate call
        # test_evaluate(
        #     global_target,
        #     instance_targets=[
        #         {"score": instance["score"]} for instance in instance_targets
        #     ],
        #     task_data=task_data,
        #     metric_name=catalog_name,
        # )

        # test using the usual metric pipeline
        test_pipeline = MetricPipeline(
            main_score=main_score,
            preprocess_steps=get_test_pipeline_task_preprocess_steps(task),
            metric=f"{catalog_name}",
        )
        short_catalog_name = catalog_name.split(".")[-1]
        instance_targets = [
            add_scores_prefix_to_target(i, short_catalog_name, dimension)
            for i in instance_targets
        ]
        global_target = add_scores_prefix_to_target(
            global_target, short_catalog_name, dimension
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
