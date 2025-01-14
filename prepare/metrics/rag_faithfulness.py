from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy, Rename
from unitxt.test_utils.metrics import test_metric

base = "metrics.rag_by_task"
default = "token_k_precision"
dimension = "faithfulness"
task_names = ["autorag", "response_generation", "end_to_end"]


def get_scores_prefix(metric_catalog_name, dim_name):
    if metric_catalog_name == dim_name:
        return f"{dim_name}_"
    return f"{dim_name}_{metric_catalog_name}_"


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
            Copy(
                field_to_field={
                    "contexts": "references",
                    "answer": "prediction",
                },
            ),
        ]
    if task == "response_generation":
        return [
            Copy(
                field_to_field={
                    "task_data/contexts": "references",
                }
            ),
        ]
    if task == "end_to_end":
        return [
            Copy(
                field_to_field={
                    "prediction/contexts": "references",
                    "prediction/answer": "prediction",
                }
            ),
        ]
    raise ValueError(f"Unsupported rag task {task}")


def get_test_pipeline_task_preprocess_steps(task):
    if task == "autorag":
        return [
            Rename(field_to_field={"task_data/contexts": "contexts"}),
            Rename(field_to_field={"task_data/answer": "answer"}),
        ]
    if task == "response_generation":
        return [
            Copy(field_to_field={"task_data/answer": "prediction"}),
        ]
    if task == "end_to_end":
        return [
            Copy(field_to_field={"task_data/answer": "prediction/answer"}),
            Copy(field_to_field={"task_data/contexts": "prediction/contexts"}),
        ]
    raise ValueError(f"Unsupported rag task {task}")


for task in task_names:
    for new_catalog_name, base_catalog_name, main_score in [
        ("token_k_precision", "metrics.token_overlap", "precision"),
        (
            "bert_score_k_precision",
            "metrics.bert_score.deberta_large_mnli",
            "precision",
        ),
        (
            "bert_score_k_precision_ml",
            "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
            "precision",
        ),
        ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
        ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
        ("vectara_hhem_2_1", "metrics.vectara_groundedness_hhem_2_1", "hhem_score"),
    ]:
        metric = MetricPipeline(
            main_score=main_score,
            preprocess_steps=get_preprocess_steps(task),
            metric=base_catalog_name,
            score_prefix=get_scores_prefix(new_catalog_name, dimension),
        )
        add_to_catalog(
            metric, f"{base}.{task}.{dimension}.{new_catalog_name}", overwrite=True
        )

        if new_catalog_name == default:
            metric = MetricPipeline(
                main_score=main_score,
                preprocess_steps=get_preprocess_steps(task),
                metric=base_catalog_name,
                score_prefix=f"{dimension}_",
            )
            add_to_catalog(metric, f"{base}.{task}.{dimension}", overwrite=True)


def test_faithfulness(
    task_data, catalog_name, global_target, instance_targets, main_score, task
):
    # print(catalog_name)
    # test the evaluate call
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
        predictions=[None] * len(instance_targets),
        references=[[]] * len(instance_targets),
        instance_targets=instance_targets,
        global_target=global_target,
        task_data=task_data,
    )


def test_faithfulness_sentence_bert(task):
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
        catalog_name=f"{base}.{task}.{dimension}.sentence_bert_bge",
        global_target={
            "score": 0.64,
            "score_ci_high": 0.75,
            "score_ci_low": 0.53,
            "sbert_score": 0.64,
            "sbert_score_ci_high": 0.75,
            "sbert_score_ci_low": 0.53,
            "score_name": "sbert_score",
            "num_of_instances": 2,
        },
        instance_targets=[
            {
                "sbert_score": 0.75,
                "score": 0.75,
                "score_name": "sbert_score",
            },
            {
                "sbert_score": 0.53,
                "score": 0.53,
                "score_name": "sbert_score",
            },
        ],
        main_score="sbert_score",
        task=task,
    )

    test_faithfulness(
        task_data,
        catalog_name=f"{base}.{task}.{dimension}.sentence_bert_mini_lm",
        global_target={
            "score": 0.17,
            "score_ci_high": 0.42,
            "score_ci_low": -0.08,
            "sbert_score": 0.17,
            "sbert_score_ci_high": 0.42,
            "sbert_score_ci_low": -0.08,
            "score_name": "sbert_score",
            "num_of_instances": 2,
        },
        instance_targets=[
            {
                "sbert_score": 0.42,
                "score": 0.42,
                "score_name": "sbert_score",
            },
            {
                "sbert_score": -0.08,
                "score": -0.08,
                "score_name": "sbert_score",
            },
        ],
        main_score="sbert_score",
        task=task,
    )


def test_faithfulness_token_k_precision(task):
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
        "num_of_instances": 2,
    }

    for catalog_name, global_target, instance_targets in [
        (
            f"{base}.{task}.{dimension}",
            precision_global_target,
            precision_instance_targets,
        ),
        (
            f"{base}.{task}.{dimension}.{default}",
            precision_global_target,
            precision_instance_targets,
        ),
    ]:
        test_faithfulness(
            task_data,
            catalog_name,
            global_target,
            instance_targets,
            main_score="precision",
            task=task,
        )


if __name__ == "__main__":
    for task in task_names:
        # This test does not involve any models
        test_faithfulness_token_k_precision(task)

        # Tests which involve models:
        test_faithfulness_sentence_bert(task)
