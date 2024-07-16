from unitxt import add_to_catalog
from unitxt.metrics import FinQAEval

metric = FinQAEval()
add_to_catalog(metric, "metrics.fin_qa_metric", overwrite=True)
