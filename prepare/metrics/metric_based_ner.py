from unitxt import add_to_catalog
from unitxt.metrics import MetricBasedNer
from unitxt.test_utils.metrics import test_metric

metric = MetricBasedNer(
    metric="metrics.accuracy",
    n_resamples=0,
    min_score_for_match=0.75,
    __description__="""
    Calculates f1 metrics for NER , by comparing entity using a provided Unitxt metric.

    This customizable metric can use any Unitxt metric to compare entities, including LLM as Judge.
    The metric must accept string prediction and references as input.  The similarity threshold is
    set by the 'min_score_for_match' attribute.

    By default it uses accuracy (exact match), but the metic can be overridden.

    metrics.metric_based_ner[metric=metrics.llm_as_judge.direct.watsonx.llama3_3_70b[criteria=metrics.llm_as_judge.direct.criteria.correctness_based_on_ground_truth,context_fields=ground_truth]]

     """,
)
# Test1 single line single class
# 0.1 simple case, multi examples
predictions = [
    [("Amir", "Person"), ("Yaron", "Person")],
    [("Ran", "Person"), ("Yonatan", "Person")],
]
references = [[[("Yaron", "Person"), ("Ran", "Person")]], [[("Yonatan", "Person")]]]

instance_targets = [
    {
        "f1_Person": 0.5,
        "f1_macro": 0.5,
        "in_classes_support": 1.0,
        "f1_micro": 0.5,
        "recall_micro": 0.5,
        "recall_macro": 0.5,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "score": 0.5,
        "score_name": "f1_micro",
    },
    {
        "f1_Person": 0.67,
        "f1_macro": 0.67,
        "in_classes_support": 1.0,
        "f1_micro": 0.67,
        "recall_micro": 1.0,
        "recall_macro": 1.0,
        "precision_micro": 0.5,
        "precision_macro": 0.5,
        "score": 0.67,
        "score_name": "f1_micro",
    },
]
global_target = {
    "f1_Person": 0.57,
    "f1_macro": 0.57,
    "in_classes_support": 1.0,
    "f1_micro": 0.57,
    "recall_micro": 0.67,
    "recall_macro": 0.67,
    "precision_micro": 0.5,
    "precision_macro": 0.5,
    "score": 0.57,
    "score_name": "f1_micro",
    "num_of_instances": 2,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)


add_to_catalog(metric, "metrics.metric_based_ner", overwrite=True)
