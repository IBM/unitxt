from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy

base = "metrics.rag.context_relevance"
default = "perplexity"

for new_catalog_name, base_catalog_name, main_score in [
    ("perplexity_flan_t5_small", "metrics.perplexity_q.flan_t5_small", "perplexity"),
    ("sentence_bert_bge", "metrics.sentence_bert.bge_large_en_1_5", "score"),
    ("sentence_bert_mini_lm", "metrics.sentence_bert.bge_large_en_1_5", "score"),
]:
    metric = MetricPipeline(
        main_score=main_score,
        preprocess_steps=[
            Copy(field="contexts", to_field="references"),
            Copy(field="question", to_field="prediction"),
        ],
        metric=base_catalog_name,
    )
    add_to_catalog(metric, f"{base}.{new_catalog_name}", overwrite=True)

    if new_catalog_name == default:
        add_to_catalog(metric, base, overwrite=True)

context_perplexity = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="question", to_field="prediction"),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
    postpreprocess_steps=[
        Copy(field="score/instance/reference_scores", to_field="score/instance/score")
    ],
)
add_to_catalog(context_perplexity, "metrics.rag.context_perplexity", overwrite=True)
