from src.unitxt.catalog import add_to_catalog
from src.unitxt.metrics import F1, F1Macro, F1Micro, F1MacroMultiLabel,F1MicroMultiLabel

metric = F1Macro()
add_to_catalog(metric, 'metrics.f1_macro', overwrite=True)

metric = F1Micro()
add_to_catalog(metric, 'metrics.f1_micro', overwrite=True)

metric = F1MacroMultiLabel()
add_to_catalog(metric, 'metrics.f1_macro_multi_label', overwrite=True)

metric = F1MicroMultiLabel()
add_to_catalog(metric, 'metrics.f1_micro_multi_label', overwrite=True)

