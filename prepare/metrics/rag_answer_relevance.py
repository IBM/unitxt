from unitxt import add_to_catalog
from unitxt.metrics import (
    MetricPipeline,
)
from unitxt.operators import Copy, ListFieldValues

answer_reward = MetricPipeline(
    main_score="score",
    preprocess_steps=[
        Copy(field="question", to_field="references"),
        Copy(field="answer", to_field="prediction"),
        # This metric compares the answer (as the prediction) to the question (as the reference).
        # We have to wrap the question by a list (otherwise it will be a string),
        # because references are expected to be lists
        ListFieldValues(fields=["references"], to_field="references"),
    ],
    metric="metrics.reward.deberta_v3_large_v2",
)
add_to_catalog(answer_reward, "metrics.rag.answer_reward", overwrite=True)
answer_inference = MetricPipeline(
    main_score="perplexity",
    preprocess_steps=[
        Copy(field="contexts", to_field="references"),
        Copy(field="answer", to_field="prediction"),
    ],
    metric="metrics.perplexity_nli.t5_nli_mixture",
)
add_to_catalog(answer_inference, "metrics.rag.answer_inference", overwrite=True)
