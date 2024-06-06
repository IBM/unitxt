from unitxt import add_to_catalog
from unitxt.metrics import HuggingfaceMetric, MetricPipeline
from unitxt.operators import Copy, MapInstanceValues
from unitxt.test_utils.metrics import test_metric

metric = MetricPipeline(
    main_score="sacrebleu",
    preprocess_steps=[
        Copy(
            field="task_data/target_language",
            to_field="task_data/tokenize",
            not_exist_ok=True,
            get_default="en",
        ),
        MapInstanceValues(
            mappers={"task_data/tokenize": {"en": "", "ja": "ja-mecab"}},
            strict=True,
        ),
    ],
    metric=HuggingfaceMetric(
        hf_metric_name="sacrebleu",
        hf_main_score="score",
        prediction_type="str",
        main_score="sacrebleu",
        scale=1.0,
        scaled_fields=["sacrebleu", "precisions"],
        hf_additional_input_fields_pass_one_value=["tokenize"],
    ),
)

predictions = ["hello there general kenobi", "on our way to ankh morpork"]
references = [
    ["hello there general kenobi", "hello there !"],
    ["goodbye ankh morpork", "ankh morpork"],
]
instance_targets = [
    {
        "counts": [4, 3, 2, 1],
        "totals": [4, 3, 2, 1],
        "precisions": [100.0, 100.0, 100.0, 100.0],
        "bp": 1.0,
        "sys_len": 4,
        "ref_len": 4,
        "sacrebleu": 100.0,
        "score": 100.0,
        "score_name": "sacrebleu",
    },
    {
        "counts": [2, 1, 0, 0],
        "totals": [6, 5, 4, 3],
        "precisions": [33.33, 20.0, 12.5, 8.33],
        "bp": 1.0,
        "sys_len": 6,
        "ref_len": 3,
        "sacrebleu": 16.23,
        "score": 16.23,
        "score_name": "sacrebleu",
    },
]

global_target = {
    "sacrebleu": 39.76,
    "score": 39.76,
    "counts": [6, 4, 2, 1],
    "totals": [10, 8, 6, 4],
    "precisions": [60.0, 50.0, 33.33, 25.0],
    "bp": 1.0,
    "sys_len": 10,
    "ref_len": 7,
    "score_name": "sacrebleu",
    "sacrebleu_ci_low": 11.48,
    "sacrebleu_ci_high": 100.0,
    "score_ci_low": 11.48,
    "score_ci_high": 100.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.sacrebleu", overwrite=True)
