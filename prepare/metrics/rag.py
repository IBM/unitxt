from unitxt import add_to_catalog
from unitxt.metrics import (
    BertScore,
    MetricPipeline,
    Reward,
    SentenceBert,
    TokenOverlap,
)
from unitxt.operators import Copy, ListFieldValues
from unitxt.serializers import MultiTypeSerializer
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
    "metrics.sentence_bert.minilm_l12_v2": SentenceBert(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    ),
    "metrics.sentence_bert.bge_large_en_1_5": SentenceBert(
        model_name="BAAI/bge-large-en-v1.5"
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
    "num_of_instances": 3,
}
metric = MetricPipeline(
    main_score="f1",
    preprocess_steps=[
        Copy(field="task_data/context", to_field="references"),
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric=metrics["metrics.token_overlap"],
    postprocess_steps=[
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
    "num_of_instances": 2,
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
    "num_of_instances": 2,
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
    "num_of_instances": 2,
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
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
    "num_of_instances": 2,
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
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
    "num_of_instances": 2,
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
metric = metrics["metrics.sentence_bert.mpnet_base_v2"]
predictions = ["hello there general dude", "foo bar foobar"]
references = [
    ["hello there general kenobi", "hello there!"],
    ["foo bar foobar", "foo bar"],
]
instance_targets = [
    {"sbert_score": 0.71, "score": 0.71, "score_name": "sbert_score"},
    {"sbert_score": 1.0, "score": 1.0, "score_name": "sbert_score"},
]
global_target = {
    "sbert_score": 0.86,
    "sbert_score_ci_high": 1.0,
    "sbert_score_ci_low": 0.71,
    "score": 0.86,
    "score_ci_high": 1.0,
    "score_ci_low": 0.71,
    "score_name": "sbert_score",
    "num_of_instances": 2,
}
# test_metric(
#     metric=metric,
#     predictions=predictions,
#     references=references,
#     instance_targets=instance_targets,
#     global_target=global_target,
# )
metric = metrics["metrics.reward.deberta_v3_large_v2"]
predictions = ["hello there General Dude", "foo bar foobar"]
references = [["How do you greet General Dude"], ["What is your name?"]]
instance_targets = [
    {
        "label": "LABEL_0",
        "score": 0.14,
        "score_name": "reward_score",
        "reward_score": 0.14,
    },
    {
        "label": "LABEL_0",
        "score": 0.03,
        "score_name": "reward_score",
        "reward_score": 0.03,
    },
]
global_target = {
    "reward_score": 0.09,
    "reward_score_ci_high": 0.14,
    "reward_score_ci_low": 0.03,
    "score": 0.09,
    "score_ci_high": 0.14,
    "score_ci_low": 0.03,
    "score_name": "reward_score",
    "num_of_instances": 2,
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

for axis, base_metric, main_score, new_metric in [
    ("correctness", "token_overlap", "f1", "answer_correctness.token_recall"),
    (
        "correctness",
        "bert_score.deberta_large_mnli",
        "recall",
        "answer_correctness.bert_score_recall",
    ),
    (
        "correctness",
        "bert_score.deberta_v3_base_mnli_xnli_ml",
        "recall",
        "answer_correctness.bert_score_recall_ml",
    ),
    ("faithfullness", "token_overlap", "precision", "faithfulness.token_k_precision"),
]:
    deprecated_path = f"metrics.rag.response_generation.{axis}.{base_metric}"
    new_metric_path = f"metrics.rag.response_generation.{new_metric}"
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
        postprocess_steps=[
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
        __deprecated_msg__=f"Metric {deprecated_path} is deprecated. Please use {new_metric_path} instead.",
    )

    add_to_catalog(
        metric,
        f"metrics.rag.response_generation.{axis}.{base_metric}",
        overwrite=True,
    )

# end to end

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

assert len(end_to_end_artifact_name_to_main_score) == len(
    end_to_end_artifact_names_to_main_metric
)

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
        MultiTypeSerializer(field="references", process_every_value=True),
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
        MultiTypeSerializer(field="prediction"),
    ],
}


for artifact_name in end_to_end_artifact_names_to_preprocess_steps.keys():
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
