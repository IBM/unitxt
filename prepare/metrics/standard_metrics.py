from src.unitxt.catalog import add_to_catalog
from src.unitxt.standard_metrics import (
    StandardAccuracy,
    StandardAccuracyMultiLabel,
    StandardF1Macro,
    StandardF1MacroMultiLabel,
    StandardF1Micro,
    StandardF1MicroMultiLabel,
)

standard_accuracy = StandardAccuracy(metric_name="metrics.accuracy")
add_to_catalog(standard_accuracy, "standard_metrics.accuracy", overwrite=True)

standard_accuracy_multi_label = StandardAccuracyMultiLabel(
    metric_name="standard_metrics.accuracy_multi_label"
)
add_to_catalog(
    standard_accuracy_multi_label,
    "standard_metrics.accuracy_multi_label",
    overwrite=True,
)

standard_f1_macro = StandardF1Macro(metric_name="metrics.f1_macro")
add_to_catalog(standard_f1_macro, "standard_metrics.f1_macro", overwrite=True)

standard_f1_micro = StandardF1Micro(metric_name="metrics.f1_micro")
add_to_catalog(standard_f1_micro, "standard_metrics.f1_micro", overwrite=True)

standard_f1_macro_multi_label = StandardF1MacroMultiLabel(
    metric_name="metrics.f1_macro_multi_label"
)
add_to_catalog(
    standard_f1_macro_multi_label,
    "standard_metrics.f1_macro_multi_label",
    overwrite=True,
)

standard_f1_micro_multi_label = StandardF1MicroMultiLabel(
    metric_name="metrics.f1_micro_multi_label"
)
add_to_catalog(
    standard_f1_micro_multi_label,
    "standard_metrics.f1_micro_multi_label",
    overwrite=True,
)
