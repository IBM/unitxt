from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    Accuracy,
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
    ("metrics.bert_score.deberta.xlarge.mnli", BertScore(model_name="microsoft/deberta-xlarge-mnli")),
    ("metrics.sentence_bert.mpnet.base.v2", SentenceBert(model_name="sentence-transformers/all-mpnet-base-v2")),
    ("metrics.reward.deberta.v3.large.v2", Reward(model_name="OpenAssistant/reward-model-deberta-v3-large-v2")),
]
for metric_id, metric in metrics:
    add_to_catalog(metric, metric_id, overwrite=True)

predictions = ["apple", "boy", "cat"]
references = [["apple2"], ["boys"], ["dogs"]]
inputs = [{"context": "apple 2e"}, {"context": "boy"}, {"context": "dog"}]
outputs = [{}, {}, {}]
instance_targets = [  # nDCG is undefined at instance level
    {"f1": 0.67, "precision": 1.0, "recall": 0.5, "score": 0.67, "score_name": "f1"},
    {"f1": 1.0, "precision": 1.0, "recall": 1.0, "score": 1.0, "score_name": "f1"},
    {"f1": 0, "precision": 0, "recall": 0, "score": 0, "score_name": "f1"},
]

global_target = {"f1": 0.56, "precision": 0.67, "recall": 0.5, "score": 0.56, "score_name": "f1"}

metric = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        CopyFields(field_to_field=[("inputs/context", "references")], use_query=True),
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    # metric=SentenceBert(model_name="sentence-transformers/all-mpnet-base-v2"),
    metric=TokenOverlap(),
)

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    inputs=inputs,
    outputs=outputs,
)

add_to_catalog(metric, "metrics.token_overlap_with_context", overwrite=True)
