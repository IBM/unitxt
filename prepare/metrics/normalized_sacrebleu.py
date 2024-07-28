from unitxt import add_to_catalog
from unitxt.metrics import MetricPipeline, NormalizedSacrebleu
from unitxt.operators import Copy, MapInstanceValues
from unitxt.processors import LowerCase
from unitxt.test_utils.metrics import test_metric

language_to_tokenizer = {
    "german": None,
    "deutch": None,
    "de": None,
    "french": None,
    "fr": None,
    "romanian": None,
    "ro": None,
    "english": None,
    "en": None,
    "spanish": None,
    "es": None,
    "portuguese": None,
    "pt": None,
    "arabic": "intl",
    "ar": "intl",
    "korean": "ko-mecab",
    "ko": "ko-mecab",
    "japanese": "ja-mecab",
    "ja": "ja-mecab",
}

metric = MetricPipeline(
    main_score="sacrebleu",
    prediction_type="str",
    preprocess_steps=[
        Copy(
            field="task_data/target_language",
            to_field="task_data/tokenize",
            not_exist_ok=True,
            get_default="en",
        ),
        LowerCase(field="task_data/tokenize"),
        MapInstanceValues(
            mappers={"task_data/tokenize": language_to_tokenizer},
            strict=True,
        ),
    ],
    metric=NormalizedSacrebleu(),
)

### ENGLISH

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


### JAPANESE

predictions = [
    "他の専門家たちと同様に、彼は糖尿病を完治できるかどうかについては懐疑的であり、これらの調査結果はすでにI型糖尿病を患っている人々には何の関連性もないことを指摘しています。",
    "他方、成績評価の甘い授業がく評価されたり、人気取に走教師が出たりし、成績のりや大学教師のレベルダウという弊害をもたら恐れがある、などの反省見もある.",
]
references = [
    [
        "他の専門家たちと同様に、彼は糖尿病を完治できるかどうかについては懐疑的であり、これらの調査結果はすでにI型糖尿病を患っている人々には何の関連性もないことを指摘しています。"
    ],
    [
        "他方、成績評価の甘い授業が高く評価されたり、人気取りに走る教師が出たりし、成績の安売りや大学教師のレベルダウンという弊害をもたらす恐れがある、などの反省意見もある."
    ],
]
task_data = len(predictions) * [{"target_language": "ja"}]

instance_targets = [
    {
        "counts": [57, 56, 55, 54],
        "totals": [57, 56, 55, 54],
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "bp": 1.0,
        "sys_len": 57,
        "ref_len": 57,
        "sacrebleu": 1.0,
        "score": 1.0,
        "score_name": "sacrebleu",
    },
    {
        "counts": [39, 31, 24, 17],
        "totals": [47, 46, 45, 44],
        "precisions": [0.83, 0.67, 0.53, 0.39],
        "bp": 0.98,
        "sys_len": 47,
        "ref_len": 48,
        "sacrebleu": 0.57,
        "score": 0.57,
        "score_name": "sacrebleu",
    },
]


global_target = {
    "counts": [96, 87, 79, 71],
    "totals": [104, 102, 100, 98],
    "precisions": [0.92, 0.85, 0.79, 0.72],
    "bp": 0.99,
    "sys_len": 104,
    "ref_len": 105,
    "sacrebleu": 0.81,
    "score": 0.81,
    "score_name": "sacrebleu",
    "score_ci_low": 0.57,
    "score_ci_high": 1.0,
    "sacrebleu_ci_low": 0.57,
    "sacrebleu_ci_high": 1.0,
}
outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)

### ARABIC

predictions = ["لى يسارك ، بر ماركت.", "ﻣَﺮَّﺕ ﻋِﺪَّﺓُ ﺳَﻨَﻮَﺍﺕٍ ﻗَﺒﻞ ﺃَﻥ ﺃَﺭَﺍﻫَﺎ ﻣِﻦ ﺟَﺪِﻳﺪٍ"]
references = [["على ، ستمر سوبر ماركت."], ["ﻣَﺮَّﺕ ﻋِﺪَّﺓُ ﺳَﻨَﻮَﺍﺕٍ ﻗَﺒﻞ ﺃَﻥ ﺃَﺭَﺍﻫَﺎ ﻣِﻦ ﺟَﺪِﻳﺪٍ"]]
task_data = len(predictions) * [{"target_language": "ar"}]
instance_targets = [
    {
        "counts": [3, 1, 0, 0],
        "totals": [6, 5, 4, 3],
        "precisions": [0.5, 0.2, 0.12, 0.08],
        "bp": 1.0,
        "sys_len": 6,
        "ref_len": 6,
        "sacrebleu": 0.18,
        "score": 0.18,
        "score_name": "sacrebleu",
    },
    {
        "counts": [8, 7, 6, 5],
        "totals": [8, 7, 6, 5],
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "bp": 1.0,
        "sys_len": 8,
        "ref_len": 8,
        "sacrebleu": 1.0,
        "score": 1.0,
        "score_name": "sacrebleu",
    },
]

global_target = {
    "counts": [11, 8, 6, 5],
    "totals": [14, 12, 10, 8],
    "precisions": [0.79, 0.67, 0.6, 0.62],
    "bp": 1.0,
    "sys_len": 14,
    "ref_len": 14,
    "sacrebleu": 0.67,
    "score": 0.67,
    "score_name": "sacrebleu",
    "score_ci_low": 0.13,
    "score_ci_high": 1.0,
    "sacrebleu_ci_low": 0.13,
    "sacrebleu_ci_high": 1.0,
}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    instance_targets=instance_targets,
    global_target=global_target,
    task_data=task_data,
)

### KOREAN

predictions = ["이게에 신을 살 거예요", "저는 한국 친구를 사귀고 싶습니다"]
references = [
    ["이 가게에서 신발을 살 거예요", "이 가에서 신발살 거예요"],
    ["저는 한국 친구를 사귀고 싶습니다", "저는 한구를 사귀 싶습니다"],
]
task_data = len(predictions) * [{"target_language": "Korean"}]

instance_targets = [
    {
        "counts": [4, 3, 2, 1],
        "totals": [7, 6, 5, 4],
        "precisions": [0.57, 0.5, 0.4, 0.25],
        "bp": 1.0,
        "sys_len": 7,
        "ref_len": 7,
        "sacrebleu": 0.41,
        "score": 0.41,
        "score_name": "sacrebleu",
    },
    {
        "counts": [9, 8, 7, 6],
        "totals": [9, 8, 7, 6],
        "precisions": [1.0, 1.0, 1.0, 1.0],
        "bp": 1.0,
        "sys_len": 9,
        "ref_len": 9,
        "sacrebleu": 1.0,
        "score": 1.0,
        "score_name": "sacrebleu",
    },
]

global_target = {
    "counts": [13, 11, 9, 7],
    "totals": [16, 14, 12, 10],
    "precisions": [0.81, 0.79, 0.75, 0.7],
    "bp": 1.0,
    "sys_len": 16,
    "ref_len": 16,
    "sacrebleu": 0.76,
    "score": 0.76,
    "score_name": "sacrebleu",
    "score_ci_low": 0.41,
    "score_ci_high": 1.0,
    "sacrebleu_ci_low": 0.41,
    "sacrebleu_ci_high": 1.0,
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
