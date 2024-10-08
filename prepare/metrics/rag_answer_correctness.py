from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Rename, Set
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
            Rename(field_to_field={"task_data/ground_truths": "ground_truths"}),
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


base = "metrics.rag.answer_correctness"
default = "token_recall"

for new_catalog_name, base_catalog_name, main_score in [
    ("token_recall", "metrics.token_overlap", "recall"),
    ("bert_score_recall", "metrics.bert_score.deberta_large_mnli", "recall"),
    (
        "bert_score_recall_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
        "recall",
    ),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "score"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="ground_truths", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
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
        metric=base_catalog_name,
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)


def test_answer_correctness_sentence_bert():
    task_data = [
        {
            # Similar sentences
            "ground_truths": ["Here is a cat."],
            "answer": "Here is a dog.",
        },
        {
            # Not so similar
            "ground_truths": ["Apples and Oranges."],
            "answer": "Here is a dog.",
        },
    ]

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.answer_correctness.sentence_bert_bge",
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

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.answer_correctness.sentence_bert_mini_lm",
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


def test_answer_correctness_token_recall(task_data):
    recall_instance_targets = [
        {
            "f1": 0.67,
            "precision": 1.0,
            "recall": 0.5,
            "score": 0.5,
            "score_name": "recall",
        },
        {
            "f1": 0.33,
            "precision": 0.33,
            "recall": 0.33,
            "score": 0.33,
            "score_name": "recall",
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
        "score_ci_high": 0.5,
        "score_ci_low": 0.33,
        "score_name": "recall",
    }

    for catalog_name, global_target, instance_targets in [
        (
            "metrics.rag.answer_correctness",
            recall_global_target,
            recall_instance_targets,
        ),
        (
            "metrics.rag.answer_correctness.token_recall",
            recall_global_target,
            recall_instance_targets,
        ),
    ]:
        test_answer_correctness(
            task_data, catalog_name, global_target, instance_targets
        )


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
# This test is here since it does not involve any models
test_answer_correctness_token_recall(task_data)

if __name__ == "__main__":
    # Tests which involve models:
    test_answer_correctness_sentence_bert()

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.answer_correctness.bert_score_recall",
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
            "score_name": "recall",
        },
        instance_targets=[
            {
                "f1": 0.71,
                "precision": 0.77,
                "recall": 0.71,
                "score": 0.71,
                "score_name": "recall",
            },
            {
                "f1": 0.71,
                "precision": 0.71,
                "recall": 0.71,
                "score": 0.71,
                "score_name": "recall",
            },
        ],
    )

    test_answer_correctness(
        task_data,
        catalog_name="metrics.rag.answer_correctness.bert_score_recall_ml",
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
            "score_name": "recall",
        },
        instance_targets=[
            {
                "f1": 0.97,
                "precision": 0.97,
                "recall": 0.97,
                "score": 0.97,
                "score_name": "recall",
            },
            {
                "f1": 0.74,
                "precision": 0.74,
                "recall": 0.74,
                "score": 0.74,
                "score_name": "recall",
            },
        ],
    )
