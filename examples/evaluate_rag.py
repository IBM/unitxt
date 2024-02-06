import sys
from ast import literal_eval

import pandas as pd

from src.unitxt.eval_utils import evaluate

if __name__ == "__main__":
    df = pd.read_csv(
        filepath_or_buffer=sys.argv[1],
        converters={
            "ground_truths": literal_eval,
            "ground_truths_context_ids": literal_eval,
            "contexts": literal_eval,
            "context_ids": literal_eval,
        },
    )

    evaluate(
        df,
        metric_names=[
            # "metrics.rag.mrr",
            # "metrics.rag.map",
            # "metrics.rag.answer_correctness",
            "metrics.rag.context_relevance",
            # "metrics.rag.faithfulness",
            # "metrics.rag.answer_reward",
            # "metrics.rag.context_correctness",
            # "metrics.rag.context_perplexity",
        ],
    ).round(2).to_csv("dataset_out.csv")

# metrics.rag.context_relevance = max over reference_scores

# "perplexity" =>
# # metrics.rag.context_relevance.max = max over reference_scores
#
# "reference_scores" =>
# metrics.rag.context_relevance.scores = [0.1, 0.2]
#
# "mean_scores_at_k" =>
# # metrics.rag.context_relevance.mean_at_k = 0.3
# # metrics.rag.context_relevance.mean_at_k
# # metrics.rag.context_relevance@5
#
# "match_at_1" =>
# # metrics.rag.retrieval_match@1
#
# "match_at_3" =>
# # metrics.rag.retrieval_at_k.score
# # metrics.rag.retrieval_at_k.match_at_1
# # metrics.rag.retrieval_at_k.match_at_3
# # metrics.rag.retrieval_at_k.match_at_5
# # metrics.rag.retrieval_at_k.match_at_10
# # metrics.rag.retrieval_at_k.match_at_20
#
# # metrics.rag.retrieval_precision@1
# # metrics.rag.retrieval_precision@3
# # metrics.rag.retrieval_precision@5
# # metrics.rag.retrieval_precision@10
# # metrics.rag.retrieval_precision@20
# # metrics.rag.retrieval_precision@40
#
#
#
