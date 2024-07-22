from unitxt import add_to_catalog
from unitxt.metrics import HuggingfaceMetric

metric = HuggingfaceMetric(
    hf_metric_name="meteor", main_score="meteor", prediction_type=str
)

add_to_catalog(metric, "metrics.meteor", overwrite=True)
