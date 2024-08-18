import pandas as pd
from unitxt.eval_utils import evaluate

test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with watsonx.ai. Watsonx.ai offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports no foundation models",
        "contexts": [
            "Supported foundation models available with watsonx.ai. Watsonx.ai offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
]
metrics_base_path = "metrics.rag"
metrics_base_names = [
    "context_relevance.context_relevance_q_c",
    "answer_relevance.answer_relevance_q_a",
    "answer_correctness.answer_correctness_q_c_a_gt",
    "answer_correctness.answer_correctness_q_a_gt",
    "correctness_holistic.correctness_holistic_q_c_a",
    "faithfulness.faithfulness_c_a",
    "faithfulness.faithfulness_q_c_a",
]
model_names = ["llama_3_1_70b_instruct", "mixtral_8x7b_instruct_v01", "gpt_4_turbo"]
use_armorm = False


if __name__ == "__main__":
    df = pd.DataFrame(test_examples)

    metric_names = [
        f"{metrics_base_path}.{metric_base_name}_judge_{model_name}"
        for metric_base_name in metrics_base_names
        for model_name in model_names
    ]

    if use_armorm:
        metric_names.extend(
            [
                "metrics.rag.correctness_holistic.correctness_holistic_q_c_a_ArmoRM",
                "metrics.rag.answer_relevance.answer_relevance_q_a_ArmoRM",
            ]
        )

    result, _ = evaluate(df, metric_names)
    result = result.round(2)
    result.to_csv("dataset_out.csv")
