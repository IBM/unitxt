from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy

base = "metrics.rag.faithfulness"

for new_catalog_name, base_catalog_name in [
    ("", "metrics.token_overlap"),
    ("k_precision", "metrics.token_overlap"),
    ("bert_k_precision", "metrics.bert_score.deberta_large_mnli"),
    ("bert_k_precision_ml", "metrics.bert_score.deberta_v3_base_mnli_xnli_ml"),
    ("bge", "metrics.sentence_bert.bge_large_en_1.5"),
]:
    metric = MetricPipeline(
        main_score="precision",
        preprocess_steps=[
            Copy(field="contexts", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
        metric=base_catalog_name,
    )
    name = ".".join([x for x in [base, new_catalog_name] if x])
    add_to_catalog(metric, name, overwrite=True)
