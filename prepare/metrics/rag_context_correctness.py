from unitxt import add_to_catalog
from unitxt.collections_operators import Wrap
from unitxt.metrics import MetricPipeline
from unitxt.operators import Copy

base = "metrics.rag.context_correctness"
default = "mrr"

for new_catalog_name, base_catalog_name, main_score in [
    ("mrr", "metrics.mrr", "score"),
    ("map", "metrics.map", "score"),
    ("retrieval_at_k", "metrics.retrieval_at_k", "score"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="context_ids", to_field="prediction"),
            Wrap(
                field="ground_truths_context_ids", inside="list", to_field="references"
            ),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)
