from src.unitxt.blocks import AddFields, AddID, CopyFields
from src.unitxt.catalog import add_to_catalog
from src.unitxt.metrics import MetricPipeline, Squad
from src.unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="f1",
    preprocess_steps=[
        AddID(),
        AddFields(
            {
                "prediction_template": {"prediction_text": "PRED", "id": "ID"},
                "reference_template": {
                    "answers": {"answer_start": [-1], "text": "REF"},
                    "id": "ID",
                },
            },
            use_deepcopy=True,
        ),
        CopyFields(
            field_to_field=[
                ["references", "reference_template/answers/text"],
                ["prediction", "prediction_template/prediction_text"],
                ["id", "prediction_template/id"],
                ["id", "reference_template/id"],
            ],
        ),
        CopyFields(
            field_to_field=[
                ["reference_template", "references"],
                ["prediction_template", "prediction"],
            ],
        ),
    ],
    metric=Squad(),
)

predictions = ["1976", "Beyoncé and", "climate change"]
references = [["1905"], ["Beyoncé and Bruno Mars"], ["climate change"]]
instance_targets = [
    {"exact_match": 0.0, "f1": 0.0, "score": 0.0, "score_name": "f1"},
    {"exact_match": 0.0, "f1": 0.67, "score": 0.67, "score_name": "f1"},
    {"exact_match": 1.0, "f1": 1.0, "score": 1.0, "score_name": "f1"},
]
global_target = {
    "exact_match": 0.33,
    "f1": 0.56,
    "score": 0.56,
    "score_name": "f1",
    "f1_ci_low": 0.0,
    "f1_ci_high": 0.89,
    "score_ci_low": 0.0,
    "score_ci_high": 0.89,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.squad", overwrite=True)
