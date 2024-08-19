from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy, RenameFields
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
            RenameFields(field_to_field={"task_data/contexts": "contexts"}),
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


if __name__ == "__main__":
    test_faithfulness_sentence_bert()
