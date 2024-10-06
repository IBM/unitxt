from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy, Rename, Set
from unitxt.test_utils.metrics import test_evaluate, test_metric

base = "metrics.rag.faithfulness"
default = "token_k_precision"

for new_catalog_name, base_catalog_name, main_score in [
    ("token_k_precision", "metrics.token_overlap", "precision"),
    ("bert_score_k_precision", "metrics.bert_score.deberta_large_mnli", "precision"),
    (
        "bert_score_k_precision_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
        "precision",
    ),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "score"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="contexts", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
        metric=base_catalog_name,
        postprocess_steps=[
            Set(fields={"score/instance/score_name": main_score}),
            Set(fields={"score/global/score_name": main_score}),
            Copy(
                field_to_field=[
                    [
                        f"score/global/{main_score}_ci_low",
                        "score/global/score_ci_low",
                    ],
                    [
                        f"score/global/{main_score}_ci_high",
                        "score/global/score_ci_high",
                    ],
                ],
            ),
        ],
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)


def test_faithfulness(task_data, catalog_name, global_target, instance_targets):
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
            Rename(field_to_field={"task_data/contexts": "contexts"}),
            Rename(field_to_field={"task_data/answer": "answer"}),
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


def test_faithfulness_sentence_bert():
    task_data = [
        {
            # Similar sentences
            "contexts": ["Here is a cat."],
            "answer": "Here is a dog.",
        },
        {
            # Not so similar
            "contexts": ["Apples and Oranges."],
            "answer": "Here is a dog.",
        },
    ]

    test_faithfulness(
        task_data,
        catalog_name="metrics.rag.faithfulness.sentence_bert_bge",
        global_target={
            "score": 0.64,
            "score_ci_high": 0.75,
            "score_ci_low": 0.53,
            "score_name": "score",
        },
        instance_targets=[
            {
                "score": 0.75,
                "score_name": "score",
            },
            {
                "score": 0.53,
                "score_name": "score",
            },
        ],
    )

    test_faithfulness(
        task_data,
        catalog_name="metrics.rag.faithfulness.sentence_bert_mini_lm",
        global_target={
            "score": 0.17,
            "score_ci_high": 0.42,
            "score_ci_low": -0.08,
            "score_name": "score",
        },
        instance_targets=[
            {
                "score": 0.42,
                "score_name": "score",
            },
            {
                "score": -0.08,
                "score_name": "score",
            },
        ],
    )


def test_faithfulness_token_k_precision():
    # don't use "A" as a token because it is considered an article and removed by the token overlap
    # metric

    task_data = [
        {  # precision is 1.0 for the first context, 0 for the second context.
            # so overall its max(1.0, 0) = 1.0
            "contexts": ["B C", "C"],
            "answer": "B",
        },
        {  # precision is 1/3
            "contexts": ["D"],
            "answer": "B C D",
        },
    ]

    precision_instance_targets = [
        {
            "f1": 0.67,
            "precision": 1.0,
            "recall": 0.5,
            "score": 1.0,
            "score_name": "precision",
        },
        {
            "f1": 0.5,
            "precision": 0.33,
            "recall": 1.0,
            "score": 0.33,
            "score_name": "precision",
        },
    ]

    precision_global_target = {
        "f1": 0.58,
        "f1_ci_high": 0.67,
        "f1_ci_low": 0.5,
        "precision": 0.67,
        "precision_ci_high": 1.0,
        "precision_ci_low": 0.33,
        "recall": 0.75,
        "recall_ci_high": 1.0,
        "recall_ci_low": 0.5,
        "score": 0.67,
        "score_ci_high": 1.0,
        "score_ci_low": 0.33,
        "score_name": "precision",
    }

    for catalog_name, global_target, instance_targets in [
        (
            base,
            precision_global_target,
            precision_instance_targets,
        ),
        (
            f"{base}.{default}",
            precision_global_target,
            precision_instance_targets,
        ),
    ]:
        test_faithfulness(task_data, catalog_name, global_target, instance_targets)


# This test is here since it does not involve any models
test_faithfulness_token_k_precision()

if __name__ == "__main__":
    # Tests which involve models:
    test_faithfulness_sentence_bert()
