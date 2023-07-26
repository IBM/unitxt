from src.unitxt.catalog import add_to_catalog
from src.unitxt.metrics import F1, F1Macro, F1Micro

metric = F1Macro()
add_to_catalog(metric, 'metrics.f1_macro', overwrite=True)


metric = F1Micro()
add_to_catalog(metric, 'metrics.f1_micro', overwrite=True)


