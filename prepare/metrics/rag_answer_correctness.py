from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy, Rename
from unitxt.test_utils.metrics import test_metric

task_names = ["autorag", "response_generation", "end_to_end"]
base = "metrics.rag_by_task"
default = "token_recall"
dimension = "answer_correctness"


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


def get_test_pipeline_task_preprocess_steps(task):
    if task == "autorag":
        return [
            Rename(field_to_field={"task_data/ground_truths": "ground_truths"}),
            Rename(field_to_field={"task_data/answer": "answer"}),
        ]
    if task == "response_generation":
        return [
            Copy(field_to_field={"task_data/answer": "prediction"}),
            Copy(
                field_to_field={
                    "task_data/ground_truths": "task_data/reference_answers"
                }
            ),
        ]
    if task == "end_to_end":
        return [
            Copy(field_to_field={"task_data/answer": "prediction/answer"}),
            Copy(
                field_to_field={
                    "task_data/ground_truths": "task_data/reference_answers"
                }
            ),
        ]
    raise ValueError(f"Unsupported rag task for {dimension}:{task}")


def test_answer_correctness(
    task_data, catalog_name, global_target, instance_targets, main_score
):
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


def get_preprocess_steps(task):
    if task == "autorag":
        return [
            Copy(
                field_to_field={
                    "ground_truths": "references",
                    "answer": "prediction",
                },
            )
        ]
    if task == "response_generation":
        return [
            Copy(
                field_to_field={
                    "task_data/reference_answers": "references",
                }
            ),
        ]
    if task == "end_to_end":
        return [
            Copy(
                field_to_field={
                    "task_data/reference_answers": "references",
                    "prediction/answer": "prediction",
                }
            ),
        ]
    raise ValueError(f"Unsupported rag task {task}")


for task in task_names:
    preprocess_steps = get_preprocess_steps(task)
    for new_catalog_name, base_catalog_name, main_score in [
        ("token_recall", "metrics.token_overlap", "recall"),
        ("bert_score_recall", "metrics.bert_score.deberta_large_mnli", "recall"),
        (
            "bert_score_recall_ml",
            "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
            "recall",
        ),
        ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "sbert_score"),
        ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "sbert_score"),
    ]:
        metric = MetricPipeline(
            main_score=main_score,
            preprocess_steps=preprocess_steps.copy(),
            metric=base_catalog_name,
            score_prefix=get_scores_prefix(new_catalog_name, dimension),
        )
        add_to_catalog(
            metric,
            f"{base}.{task}.{dimension}.{new_catalog_name}",
            overwrite=True,
        )

        if new_catalog_name == default:
            metric = MetricPipeline(
                main_score=main_score,
                preprocess_steps=preprocess_steps.copy(),
                metric=base_catalog_name,
                score_prefix=f"{dimension}_",
            )
            add_to_catalog(metric, f"{base}.{task}.{dimension}", overwrite=True)


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
    )

    test_answer_correctness(
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
        "num_of_instances": 2,
    }

    for catalog_name, global_target, instance_targets in [
        (
            f"{base}.{task}.{dimension}",
            recall_global_target,
            recall_instance_targets,
        ),
        (
            f"{base}.{task}.{dimension}.token_recall",
            recall_global_target,
            recall_instance_targets,
        ),
    ]:
        test_answer_correctness(
            task_data,
            catalog_name,
            global_target,
            instance_targets,
            main_score="recall",
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


if __name__ == "__main__":
    # Tests which involve models:
    test_answer_correctness_sentence_bert()
    for task in task_names:
        test_answer_correctness_token_recall(task_data)

        test_answer_correctness(
            task_data,
            catalog_name=f"{base}.{task}.{dimension}.bert_score_recall",
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
                "num_of_instances": 2,
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
            main_score="recall",
        )

        test_answer_correctness(
            task_data,
            catalog_name=f"{base}.{task}.{dimension}.bert_score_recall_ml",
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
                "num_of_instances": 2,
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
            main_score="recall",
        )
