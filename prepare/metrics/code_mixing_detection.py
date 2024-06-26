from unitxt import add_to_catalog
from unitxt.metrics import IsCodeMixed
from unitxt.test_utils.metrics import test_metric

examples = [
    "You say goodbye, and I say hello",
    "Hello how are you won't you tell me your name?",
    "Io ho un biglietto",
    "Io ho un ticket a Roma and also un car",
    "Guyzz 1m likes vara varaikum vitraadheenga...",
    "Supper dhanush Anna mass waiting asuran",
    "Vaa thalaiva via diwali mass ur movie bikil  out",
    "أحتاج إلى switch خطة الدفع",
    "من باید برنامه پرداخت خود را تغییر دهم",
]

gold_labels = [0, 0, 0, 1, 1, 1, 1, 1, 0]
predictions = [0, 1, 0, 1, 1, 1, 1, 0, 0]  # current predictions with Starling model
instance_targets = [
    {"is_code_mixed": pred, "score": pred, "score_name": "is_code_mixed"}
    for pred in predictions
]
global_target = {
    "is_code_mixed": 0.56,
    "is_code_mixed_ci_high": 0.89,
    "is_code_mixed_ci_low": 0.22,
    "score": 0.56,
    "score_ci_high": 0.89,
    "score_ci_low": 0.22,
    "score_name": "is_code_mixed",
}
metric = IsCodeMixed()
outputs = test_metric(
    metric=metric,
    predictions=examples,
    references=[[""] for _ in examples],
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.is_code_mixed", overwrite=True)
