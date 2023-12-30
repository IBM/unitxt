from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    BertScore,
    MetricPipeline,
    Reward,
    SentenceBert,
    TokenOverlap,
)
from src.unitxt.operators import CopyFields, ListFieldValues
from src.unitxt.test_utils.metrics import test_metric

metrics = [
    (
        "metrics.token_overlap",
        TokenOverlap(),
    ),
    (
        "metrics.bert_score.deberta.xlarge.mnli",
        BertScore(model_name="microsoft/deberta-xlarge-mnli"),
    ),
    (
        "metrics.sentence_bert.mpnet.base.v2",
        SentenceBert(model_name="sentence-transformers/all-mpnet-base-v2"),
    ),
    (
        "metrics.reward.deberta.v3.large.v2",
        Reward(model_name="OpenAssistant/reward-model-deberta-v3-large-v2"),
    ),
]
for metric_id, metric in metrics:
    add_to_catalog(metric, metric_id, overwrite=True)

predictions = ["apple", "boy", "cat"]
references = [["apple2"], ["boys"], ["dogs"]]
additional_inputs = [{"context": "apple 2e"}, {"context": "boy"}, {"context": "dog"}]
instance_targets = [
    {"f1": 0.67, "precision": 1.0, "recall": 0.5, "score": 0.67, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
    {"f1": 0, "precision": 0, "recall": 0, "score": 0, "score_name": "f1"},
]

# Currently CopyFields does not delete the source fields,
# so we get both "precision" and "precision_overlap_with_context" in results

global_target = {
    "f1": 0.56,
    "f1_ci_high": 0.89,
    "f1_ci_low": 0.0,
    "f1_overlap_with_context": 0.56,
    "precision": 0.67,
    "precision_overlap_with_context": 0.67,
    "recall": 0.5,
    "recall_overlap_with_context": 0.5,
    "score": 0.56,
    "score_ci_high": 0.89,
    "score_ci_low": 0.0,
    "score_name": "f1",
}
metric = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        CopyFields(
            field_to_field=[("additional_inputs/context", "references")], use_query=True
        ),
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric=TokenOverlap(),
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
            use_query=True,
        ),
    ],
)

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    additional_inputs=additional_inputs,
)

add_to_catalog(metric, "metrics.token_overlap_with_context", overwrite=True)
