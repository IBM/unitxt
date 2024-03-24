from src.unitxt import add_to_catalog
from src.unitxt.metrics import HuggingfaceMetric, MetricPipeline
from src.unitxt.operators import CopyFields, MapInstanceValues
from src.unitxt.test_utils.metrics import test_metric

language_to_tokenizer = {
    "deutch": None,
    "french": None,
    "romanian": None,
    "english": None,
    "German": None,
    "French": None,
    "Spanish": None,
    "Portuguese": None,
    "Arabic": None,
    "Korean": None,
    "fr": None,
    "de": None,
    "es": None,
    "pt": None,
    "en": None,
    "ar": None,
    "ko": None,
    "japanese": "ja-mecab",
    "Japanese": "ja-mecab",
    "ja": "ja-mecab",
}

metric = MetricPipeline(
    main_score="sacrebleu",
    preprocess_steps=[
        CopyFields(
            field_to_field=[("task_data/target_language", "task_data/tokenize")],
            use_query=True,
            not_exist_ok=True,
            get_default="en",
        ),
        MapInstanceValues(
            mappers={"task_data/tokenize": language_to_tokenizer},
            strict=True,
            use_query=True,
        ),
    ],
    metric=HuggingfaceMetric(
        hf_metric_name="sacrebleu",
        hf_main_score="score",
        prediction_type="str",
        main_score="sacrebleu",
        scale=100.0,
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
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "bp": 1.0,
        "sys_len": 4,
        "ref_len": 4,
        "sacrebleu": 1.0,
        "score": 1.0,
        "score_name": "sacrebleu",
    },
    {
        "counts": [2, 1, 0, 0],
        "totals": [6, 5, 4, 3],
        "precisions": [0.33, 0.20, 0.12, 0.08],
        "bp": 1.0,
        "sys_len": 6,
        "ref_len": 3,
        "sacrebleu": 0.16,
        "score": 0.16,
        "score_name": "sacrebleu",
    },
]

global_target = {
    "sacrebleu": 0.4,
    "score": 0.4,
    "counts": [6, 4, 2, 1],
    "totals": [10, 8, 6, 4],
    "precisions": [0.60, 0.50, 0.33, 0.25],
    "bp": 1.0,
    "sys_len": 10,
    "ref_len": 7,
    "score_name": "sacrebleu",
    "score_ci_low": 0.11,
    "score_ci_high": 1.0,
    "sacrebleu_ci_low": 0.11,
    "sacrebleu_ci_high": 1.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
)

predictions = [
    "他の専門家たちと同様に、彼は糖尿病を完治できるかどうかについては懐疑的であり、これらの調査結果はすでにI型糖尿病を患っている人々には何の関連性もないことを指摘しています。"
]
references = [
    [
        "他の専門家たちと同様に、彼は糖尿病を完治できるかどうかについては懐疑的であり、これらの調査結果はすでにI型糖尿病を患っている人々には何の関連性もないことを指摘しています。"
    ]
]
task_data = [{"target_language": "ja"}]
instance_targets = [
    {
        "bp": 1.0,
        "counts": [57, 56, 55, 54],
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "ref_len": 57,
        "sacrebleu": 1.0,
        "score": 1.0,
        "score_name": "sacrebleu",
        "sys_len": 57,
        "totals": [57, 56, 55, 54],
    },
]

global_target = {
    "bp": 1.0,
    "counts": [57, 56, 55, 54],
    "precisions": [1.0, 1.0, 1.0, 1.0],
    "ref_len": 57,
    "sacrebleu": 1.0,
    "score": 1.0,
    "score_name": "sacrebleu",
    "sys_len": 57,
    "totals": [57, 56, 55, 54],
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)


add_to_catalog(metric, "metrics.normalized_sacrebleu", overwrite=True)
