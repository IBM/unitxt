from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy

context_relevance_perplexity = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="question", to_field="prediction"),
    ],
    metric="metrics.perplexity_q.flan_t5_small",
)
add_to_catalog(
    context_relevance_perplexity, "metrics.rag.context_relevance", overwrite=True
)

context_relevance_bge = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="question", to_field="prediction"),
    ],
    metric="metrics.sentence_bert.bge_large_en_1.5",
)
add_to_catalog(
    context_relevance_bge, "metrics.rag.context_relevance.bge", overwrite=True
)

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
