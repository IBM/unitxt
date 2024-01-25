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
            "metrics.rag.mrr",
            "metrics.rag.map",
            "metrics.rag.answer_correctness",
            "metrics.rag.context_relevance",
            "metrics.rag.faithfulness",
            "metrics.rag.answer_relevance",
            "metrics.rag.context_perplexity",
        ],
    ).round(2).to_csv("dataset_out.csv")
