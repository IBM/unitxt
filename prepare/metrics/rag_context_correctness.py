from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy

for metric_name, catalog_name in [
    ("map", "metrics.rag.map"),
    ("mrr", "metrics.rag.mrr"),
    ("mrr", "metrics.rag.context_correctness"),
    ("retrieval_at_k", "metrics.rag.retrieval_at_k"),
]:
    metric = MetricPipeline(
        main_score="score",
        preprocess_steps=[
            Copy(field="context_ids", to_field="prediction"),
            Wrap(
                field="ground_truths_context_ids", inside="list", to_field="references"
            ),
        ],
        metric=f"metrics.{metric_name}",
    )
    add_to_catalog(metric, catalog_name, overwrite=True)
