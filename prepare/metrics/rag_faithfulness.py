from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy

base = "metrics.rag.faithfulness"
default = "token_k_precision"

for new_catalog_name, base_catalog_name, main_score in [
    ("token_k_precision", "metrics.token_overlap", "precision"),
    ("bert_score_k_precision", "metrics.bert_score.deberta_large_mnli", "precision"),
    (
        "bert_score_k_precision_ml",
        "metrics.bert_score.deberta_v3_base_mnli_xnli_ml",
        "precision",
    ),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.minilm_l12_v2", "score"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="contexts", to_field="references"),
            Copy(field="answer", to_field="prediction"),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)
