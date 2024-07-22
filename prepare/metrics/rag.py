from unitxt import add_to_catalog
from unitxt.metrics import (
    BertScore,
    MetricPipeline,
    Reward,
    SentenceBert,
    TokenOverlap,
)
from unitxt.operators import Copy, ListFieldValues
from unitxt.test_utils.metrics import test_metric

metrics = {
    "metrics.token_overlap": TokenOverlap(),
    "metrics.bert_score.deberta_xlarge_mnli": BertScore(
        model_name="microsoft/deberta-xlarge-mnli"
    ),
    "metrics.bert_score.deberta_large_mnli": BertScore(
        model_name="microsoft/deberta-large-mnli"
    ),
    "metrics.bert_score.deberta_base_mnli": BertScore(
        model_name="microsoft/deberta-base-mnli"
    ),
    "metrics.bert_score.distilbert_base_uncased": BertScore(
        model_name="distilbert-base-uncased"
    ),
    "metrics.bert_score.deberta_v3_base_mnli_xnli_ml": BertScore(
        model_name="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", model_layer=10
    ),
    "metrics.bert_score.bert_base_uncased": BertScore(model_name="bert-base-uncased"),
    "metrics.sentence_bert.mpnet_base_v2": SentenceBert(
        model_name="sentence-transformers/all-mpnet-base-v2"
    ),
    "metrics.reward.deberta_v3_large_v2": Reward(
        model_name="OpenAssistant/reward-model-deberta-v3-large-v2"
    ),
}
predictions = ["apple", "boy", "cat"]
references = [["apple2"], ["boys"], ["dogs"]]
task_data = [{"context": "apple 2e"}, {"context": "boy"}, {"context": "dog"}]
instance_targets = [
    {"f1": 0.67, "precision": 1.0, "recall": 0.5, "score": 0.67, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
    {"f1": 0, "precision": 0, "recall": 0, "score": 0, "score_name": "f1"},
]
# Currently, Copy does not delete the source fields,
# so we get both "precision" and "precision_overlap_with_context" in results
global_target = {
    "f1": 0.56,
    "f1_ci_high": 0.89,
    "f1_ci_low": 0.0,
    "f1_overlap_with_context": 0.56,
    "precision": 0.67,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.0,
    "precision_overlap_with_context": 0.67,
    "recall": 0.5,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.0,
    "recall_overlap_with_context": 0.5,
    "score": 0.56,
    "score_ci_high": 0.89,
    "score_ci_low": 0.0,
    "score_name": "f1",
}
metric = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="task_data/context", to_field="references"),
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric=metrics["metrics.token_overlap"],
    postpreprocess_steps=[
        Copy(
            field_to_field=[
                ("score/global/f1", "score/global/f1_overlap_with_context"),
                ("score/global/recall", "score/global/recall_overlap_with_context"),
                (
                    "score/global/precision",
                    "score/global/precision_overlap_with_context",
                ),
            ],
        ),
    ],
)
add_to_catalog(metric, "metrics.token_overlap_with_context", overwrite=True)
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)
metric = metrics["metrics.bert_score.deberta_xlarge_mnli"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"f1": 0.8, "precision": 0.86, "recall": 0.84, "score": 0.8, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "f1": 0.9,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.8,
    "precision": 0.93,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.86,
    "recall": 0.92,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.84,
    "score": 0.9,
    "score_ci_high": 1.0,
    "score_ci_low": 0.8,
    "score_name": "f1",
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
metric = metrics["metrics.bert_score.deberta_large_mnli"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"f1": 0.73, "precision": 0.83, "recall": 0.79, "score": 0.73, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "f1": 0.87,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.73,
    "precision": 0.92,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.83,
    "recall": 0.9,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.79,
    "score": 0.87,
    "score_ci_high": 1.0,
    "score_ci_low": 0.73,
    "score_name": "f1",
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
metric = metrics["metrics.bert_score.deberta_base_mnli"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"f1": 0.81, "precision": 0.85, "recall": 0.81, "score": 0.81, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "f1": 0.9,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.81,
    "precision": 0.93,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.85,
    "recall": 0.91,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.81,
    "score": 0.9,
    "score_ci_high": 1.0,
    "score_ci_low": 0.81,
    "score_name": "f1",
}
test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
metric = metrics["metrics.bert_score.distilbert_base_uncased"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"f1": 0.85, "precision": 0.91, "recall": 0.86, "score": 0.85, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "f1": 0.93,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.85,
    "precision": 0.95,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.91,
    "recall": 0.93,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.86,
    "score": 0.93,
    "score_ci_high": 1.0,
    "score_ci_low": 0.85,
    "score_name": "f1",
}
test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
metric = metrics["metrics.bert_score.deberta_v3_base_mnli_xnli_ml"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"f1": 0.74, "precision": 0.81, "recall": 0.71, "score": 0.74, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "f1": 0.87,
    "f1_ci_high": 1.0,
    "f1_ci_low": 0.74,
    "precision": 0.91,
    "precision_ci_high": 1.0,
    "precision_ci_low": 0.81,
    "recall": 0.86,
    "recall_ci_high": 1.0,
    "recall_ci_low": 0.71,
    "score": 0.87,
    "score_ci_high": 1.0,
    "score_ci_low": 0.74,
    "score_name": "f1",
}
test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
metric = metrics["metrics.sentence_bert.mpnet_base_v2"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"score": 0.71, "score_name": "score"},
    {"score": 1.0, "score_name": "score"},
]
global_target = {
    "score": 0.86,
    "score_ci_high": 1.0,
    "score_ci_low": 0.71,
    "score_name": "score",
}
test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)
metric = metrics["metrics.reward.deberta_v3_large_v2"]
predictions = ["hello there General Dude", "foo bar foobar"]
references = [["How do you greet General Dude"], ["What is your name?"]]
instance_targets = [
    {"label": "LABEL_0", "score": 0.14, "score_name": "score"},
    {"label": "LABEL_0", "score": 0.03, "score_name": "score"},
]
global_target = {
    "score": 0.09,
    "score_ci_high": 0.14,
    "score_ci_low": 0.03,
    "score_name": "score",
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
for metric_id, metric in metrics.items():
    add_to_catalog(metric, metric_id, overwrite=True)
# rag metrics:
# reference-less:
#   context-relevance:
#       metrics.rag.context_relevance
#       metrics.rag.context_perplexity
#   faithfulness:
#       metrics.rag.faithfulness
#       metrics.rag.k_precision
#       metrics.rag.bert_k_precision        k_precision stands for "knowledge precision"
#   answer-relevance:
#       metrics.rag.answer_relevance
# reference-based:
#   context-correctness:
#       metrics.rag.mrr
#       metrics.rag.map
#       metrics.rag.precision@K
#   answer-correctness:
#       metrics.rag.correctness
#       metrics.rag.recall
#       metrics.rag.bert_recall
context_relevance = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="question", to_field="prediction"),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
)
add_to_catalog(context_relevance, "metrics.rag.context_relevance", overwrite=True)
context_perplexity = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="question", to_field="prediction"),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
    postpreprocess_steps=[
        Copy(field="score/instance/reference_scores", to_field="score/instance/score")
    ],
)
add_to_catalog(context_perplexity, "metrics.rag.context_perplexity", overwrite=True)
for new_catalog_name, base_catalog_name in [
    ("metrics.rag.faithfulness", "metrics.token_overlap"),
    ("metrics.rag.k_precision", "metrics.token_overlap"),
    ("metrics.rag.bert_k_precision", "metrics.bert_score.deberta_large_mnli"),
    (
        "metrics.rag.bert_k_precision_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
    ),
]:
    metric = MetricPipeline(
        main_score="precision",
        preprocess_steps=[
            Copy(field="contexts", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, new_catalog_name, overwrite=True)

answer_reward = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="question", to_field="references"),
        Copy(field="answer", to_field="prediction"),
        # This metric compares the answer (as the prediction) to the question (as the reference).
        # We have to wrap the question by a list (otherwise it will be a string),
        # because references are expected to be lists
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric="metrics.reward.deberta_v3_large_v2",
)
add_to_catalog(answer_reward, "metrics.rag.answer_reward", overwrite=True)
answer_inference = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="answer", to_field="prediction"),
    ],
    metric="metrics.perplexity_nli.t5_nli_mixture",
)
add_to_catalog(answer_inference, "metrics.rag.answer_inference", overwrite=True)

for axis, base_metric, main_score in [
    ("correctness", "token_overlap", "f1"),
    ("correctness", "bert_score.deberta_large_mnli", "recall"),
    ("correctness", "bert_score.deberta_v3_base_mnli_xnli_ml", "recall"),
    ("faithfullness", "token_overlap", "precision"),
]:
    preprocess_steps = (
        [
            Copy(field="task_data/contexts", to_field="references"),
        ]
        if axis == "faithfullness"
        else []
    )

    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=preprocess_steps,
        postpreprocess_steps=[
            Copy(
                field_to_field={
                    "score/instance/f1": f"score/instance/{axis}_f1_{base_metric}",
                    "score/instance/recall": f"score/instance/{axis}_recall_{base_metric}",
                    "score/instance/precision": f"score/instance/{axis}_precision_{base_metric}",
                    "score/global/f1": f"score/global/{axis}_f1_{base_metric}",
                    "score/global/recall": f"score/global/{axis}_recall_{base_metric}",
                    "score/global/precision": f"score/global/{axis}_precision_{base_metric}",
                },
                not_exist_ok=True,
            ),
        ],
        metric=f"metrics.{base_metric}",
        prediction_type=str,
    )

    add_to_catalog(
        metric, f"metrics.rag.response_generation.{axis}.{base_metric}", overwrite=True
    )

# end to end


class TaskRagEndToEndConstants:
    # Templates
    TEMPLATE_RAG_END_TO_END_JSON_PREDICTIONS: str = (
        "templates.rag.end_to_end.json_predictions"
    )
    # Task names
    TASKS_RAG_END_TO_END: str = "tasks.rag.end_to_end"
    # Metrics

    # METRICS_RAG_END_TO_END_ANSWER_CORRECTNESS: str = "metrics.rag.end_to_end.answer_correctness"
    # METRICS_RAG_END_TO_END_ANSWER_REWARD: str = "metrics.rag.end_to_end.answer_reward"
    # METRICS_RAG_END_TO_END_ANSWER_FAITHFULNESS: str = "metrics.rag.end_to_end.answer_faithfulness"
    # METRICS_RAG_END_TO_END_CONTEXT_CORRECTNESS: str = "metrics.rag.end_to_end.context_correctness"
    # METRICS_RAG_END_TO_END_CONTEXT_RELEVANCE: str = "metrics.rag.end_to_end.context_relevance"


end_to_end_artifact_names = [
    "metrics.rag.end_to_end.answer_correctness",
    "metrics.rag.end_to_end.answer_reward",
    "metrics.rag.end_to_end.answer_faithfulness",
    "metrics.rag.end_to_end.context_correctness",
    "metrics.rag.end_to_end.context_relevance",
]

end_to_end_artifact_name_to_main_score = {
    "metrics.rag.end_to_end.answer_correctness": "recall",
    "metrics.rag.end_to_end.answer_reward": "score",
    "metrics.rag.end_to_end.answer_faithfulness": "precision",
    "metrics.rag.end_to_end.context_correctness": "score",
    "metrics.rag.end_to_end.context_relevance": "score",
}

end_to_end_artifact_names_to_main_metric = {
    "metrics.rag.end_to_end.answer_correctness": "metrics.token_overlap",
    "metrics.rag.end_to_end.answer_reward": "metrics.reward.deberta_v3_large_v2",
    "metrics.rag.end_to_end.answer_faithfulness": "metrics.token_overlap",
    "metrics.rag.end_to_end.context_correctness": "metrics.mrr",
    "metrics.rag.end_to_end.context_relevance": "metrics.perplexity_q.flan_t5_small",
}

assert len(end_to_end_artifact_names) == len(end_to_end_artifact_name_to_main_score)
assert len(end_to_end_artifact_names) == len(end_to_end_artifact_names_to_main_metric)

copy_field_prediction_answer_to_prediction = Copy(
    field_to_field=[
        (
            "prediction/answer",
            "prediction",
        )
    ],
)

copy_field_reference_answers_to_references = Copy(
    field_to_field={"task_data/reference_answers": "references"},
)

copy_field_reference_contexts_to_references = Copy(
    field_to_field={"task_data/reference_contexts": "references"}
)

copy_field_prediction_contexts_to_prediction = Copy(
    field_to_field=[
        (
            "prediction/contexts",
            "prediction",
        )
    ],
)

copy_field_prediction_context_ids_to_prediction = Copy(
    field_to_field=[
        (
            "prediction/context_ids",
            "prediction",
        )
    ],
)

copy_field_reference_context_ids_to_references_in_a_list = ListFieldValues(
    fields=["task_data/reference_context_ids"],
    to_field="references",
)

copy_field_prediction_contexts_to_references = Copy(
    field_to_field=[
        (
            "prediction/contexts",
            "references",
        )
    ],
)


copy_field_question_to_prediction = Copy(
    field_to_field=[
        (
            "task_data/question",
            "prediction",
        )
    ],
)

copy_field_question_to_references_in_a_list = ListFieldValues(
    fields=["task_data/question"],
    to_field="references",
)

end_to_end_artifact_names_to_preprocess_steps = {
    "metrics.rag.end_to_end.answer_correctness": [
        copy_field_prediction_answer_to_prediction,
        copy_field_reference_answers_to_references,
    ],
    "metrics.rag.end_to_end.answer_reward": [
        copy_field_prediction_answer_to_prediction,
        copy_field_question_to_references_in_a_list,
    ],
    "metrics.rag.end_to_end.answer_faithfulness": [
        copy_field_prediction_contexts_to_references,
        copy_field_prediction_answer_to_prediction,
    ],
    "metrics.rag.end_to_end.context_correctness": [
        copy_field_prediction_context_ids_to_prediction,
        copy_field_reference_context_ids_to_references_in_a_list,
    ],
    "metrics.rag.end_to_end.context_relevance": [
        copy_field_prediction_contexts_to_references,
        copy_field_question_to_prediction,
    ],
}

assert len(end_to_end_artifact_names) == len(
    end_to_end_artifact_names_to_preprocess_steps
)

for artifact_name in end_to_end_artifact_names:
    metric_short_name = artifact_name.split(".")[-1]
    if metric_short_name == "rouge":  # rouge does not need a prefix
        score_prefix = ""
    else:
        score_prefix = f"[score_prefix={metric_short_name}_]"

    metric = MetricPipeline(
        main_score=end_to_end_artifact_name_to_main_score[artifact_name],
        preprocess_steps=end_to_end_artifact_names_to_preprocess_steps[artifact_name],
        metric=f"{end_to_end_artifact_names_to_main_metric[artifact_name]}{score_prefix}",
    )

    add_to_catalog(metric, artifact_name, overwrite=True)
