from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, RenameFields
from unitxt.test_utils.metrics import test_evaluate, test_metric


def test_answer_correctness(task_data, catalog_name, global_target, instance_targets):
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
            RenameFields(field_to_field={"task_data/ground_truths": "ground_truths"}),
            RenameFields(field_to_field={"task_data/answer": "answer"}),
        ],
        metric=f"{catalog_name}",
    )
    test_metric(
        metric=test_pipeline,
        predictions=[None] * len(instance_targets),
        references=[[]] * len(instance_targets),
        instance_targets=instance_targets,
        global_target=global_target,
        task_data=task_data,
    )


for new_catalog_name, base_catalog_name in [
    ("metrics.rag.answer_correctness", "metrics.token_overlap"),
    ("metrics.rag.recall", "metrics.token_overlap"),
    ("metrics.rag.bert_recall", "metrics.bert_score.deberta_large_mnli"),
    ("metrics.rag.bert_recall_ml", "metrics.bert_score.deberta_v3_base_mnli_xnli_ml"),
]:
    metric = MetricPipeline(
        main_score="recall",
        preprocess_steps=[
            Copy(field="ground_truths", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, new_catalog_name, overwrite=True)

if __name__ == "__main__":
    # don't use "A" as a token because it is considered an article and removed by the token overlap
    # metric
    task_data = [
        {  # recall is 0.5 for the first ground_truth, 0 for the second ground_truth.
            # so overall its max(0.5, 0) = 0.5
            "ground_truths": ["B C", "C"],
            "answer": "B",
        },
        {  # recall is 1/3
            "ground_truths": ["D E F"],
            "answer": "B C D",
        },
    ]

    recall_instance_targets = [
        {"f1": 0.67, "precision": 1.0, "recall": 0.5, "score": 0.5, "score_name": "f1"},
        {
            "f1": 0.33,
            "precision": 0.33,
            "recall": 0.33,
            "score": 0.33,
            "score_name": "f1",
        },
    ]

    recall_global_target = {
        "f1": 0.5,
        "f1_ci_high": 0.67,
        "f1_ci_low": 0.33,
        "precision": 0.67,
        "precision_ci_high": 1.0,
        "precision_ci_low": 0.33,
        "recall": 0.42,
        "recall_ci_high": 0.5,
        "recall_ci_low": 0.33,
        "score": 0.42,
        "score_ci_high": 0.67,
        "score_ci_low": 0.33,
        "score_name": "f1",
    }

    for catalog_name, global_target, instance_targets in [
        (
            "metrics.rag.answer_correctness",
            recall_global_target,
            recall_instance_targets,
        ),
        ("metrics.rag.recall", recall_global_target, recall_instance_targets),
    ]:
        test_answer_correctness(
            task_data, catalog_name, global_target, instance_targets
        )

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.bert_recall",
        global_target={
            "f1": 0.71,
            "f1_ci_high": 0.71,
            "f1_ci_low": 0.71,
            "precision": 0.74,
            "precision_ci_high": 0.77,
            "precision_ci_low": 0.71,
            "recall": 0.71,
            "recall_ci_high": 0.71,
            "recall_ci_low": 0.71,
            "score": 0.71,
            "score_ci_high": 0.71,
            "score_ci_low": 0.71,
            "score_name": "f1",
        },
        instance_targets=[
            {
                "f1": 0.71,
                "precision": 0.77,
                "recall": 0.71,
                "score": 0.71,
                "score_name": "f1",
            },
            {
                "f1": 0.71,
                "precision": 0.71,
                "recall": 0.71,
                "score": 0.71,
                "score_name": "f1",
            },
        ],
    )

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.bert_recall_ml",
        global_target={
            "f1": 0.86,
            "f1_ci_high": 0.97,
            "f1_ci_low": 0.74,
            "precision": 0.86,
            "precision_ci_high": 0.97,
            "precision_ci_low": 0.74,
            "recall": 0.86,
            "recall_ci_high": 0.97,
            "recall_ci_low": 0.74,
            "score": 0.86,
            "score_ci_high": 0.97,
            "score_ci_low": 0.74,
            "score_name": "f1",
        },
        instance_targets=[
            {
                "f1": 0.97,
                "precision": 0.97,
                "recall": 0.97,
                "score": 0.97,
                "score_name": "f1",
            },
            {
                "f1": 0.74,
                "precision": 0.74,
                "recall": 0.74,
                "score": 0.74,
                "score_name": "f1",
            },
        ],
    )
