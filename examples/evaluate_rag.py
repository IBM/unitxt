import json
import sys
from ast import literal_eval

import pandas as pd
from unitxt.eval_utils import evaluate

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

    # passing list of dicts
    result, _ = evaluate(
        df.to_dict("records"),
        metric_names=[
            "metrics.rag.mrr",
            "metrics.rag.map",
            "metrics.rag.answer_correctness",
        ],
    )
    with open("dataset_out.json", "w") as f:
        json.dump(result, f, indent=4)

    result, _ = evaluate(
        df,
        metric_names=[
            "metrics.rag.mrr",
            "metrics.rag.map",
            "metrics.rag.answer_correctness",
            "metrics.rag.context_relevance",
            "metrics.rag.faithfulness",
            "metrics.rag.answer_reward",
            "metrics.rag.context_correctness",
            "metrics.rag.context_perplexity",
        ],
    )
    result.round(2).to_csv("dataset_out.csv")
