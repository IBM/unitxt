from unitxt import add_to_catalog
from unitxt.metrics import (
    BertScore,
    MetricPipeline,
    Reward,
    SentenceBert,
    TokenOverlap,
)
from unitxt.operators import CopyFields, ListFieldValues
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
# Currently, CopyFields does not delete the source fields,
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
        CopyFields(field_to_field=[("task_data/context", "references")]),
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric=metrics["metrics.token_overlap"],
    postpreprocess_steps=[
        CopyFields(
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
for metric_name, catalog_name in [
    ("map", "metrics.rag.map"),
    ("mrr", "metrics.rag.mrr"),
    ("mrr", "metrics.rag.context_correctness"),
]:
    metric = MetricPipeline(
        main_score="score",
        preprocess_steps=[
            CopyFields(field_to_field=[("context_ids", "prediction")]),
            CopyFields(
                field_to_field=[("ground_truths_context_ids", "references")],
            ),
        ],
        metric=f"metrics.{metric_name}",
    )
    add_to_catalog(metric, catalog_name, overwrite=True)
context_relevance = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        CopyFields(field_to_field=[("contexts", "references")]),
        CopyFields(
            field_to_field=[("question", "prediction")],
        ),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
)
add_to_catalog(context_relevance, "metrics.rag.context_relevance", overwrite=True)
context_perplexity = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        CopyFields(field_to_field=[("contexts", "references")]),
        CopyFields(
            field_to_field=[("question", "prediction")],
        ),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
    postpreprocess_steps=[
        CopyFields(
            field_to_field=[
                ("score/instance/reference_scores", "score/instance/score")
            ],
        )
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
            CopyFields(field_to_field=[("contexts", "references")]),
            CopyFields(
                field_to_field=[("answer", "prediction")],
            ),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, new_catalog_name, overwrite=True)
for new_catalog_name, base_catalog_name in [
    ("metrics.rag.answer_correctness", "metrics.token_overlap"),
    ("metrics.rag.recall", "metrics.token_overlap"),
    ("metrics.rag.bert_recall", "metrics.bert_score.deberta_large_mnli"),
    ("metrics.rag.bert_recall_ml", "metrics.bert_score.deberta_v3_base_mnli_xnli_ml"),
]:
    metric = MetricPipeline(
        main_score="recall",
        preprocess_steps=[
            CopyFields(field_to_field=[("ground_truths", "references")]),
            CopyFields(
                field_to_field=[("answer", "prediction")],
            ),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, new_catalog_name, overwrite=True)
answer_reward = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        CopyFields(field_to_field=[("question", "references")]),
        CopyFields(
            field_to_field=[("answer", "prediction")],
        ),
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
        CopyFields(field_to_field=[("contexts", "references")]),
        CopyFields(
            field_to_field=[("answer", "prediction")],
        ),
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
            CopyFields(field_to_field=[("task_data/contexts", "references")]),
        ]
        if axis == "faithfullness"
        else []
    )

    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=preprocess_steps,
        postpreprocess_steps=[
            CopyFields(
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
        prediction_type="str",
    )

    add_to_catalog(
        metric, f"metrics.rag.response_generation.{axis}.{base_metric}", overwrite=True
    )
