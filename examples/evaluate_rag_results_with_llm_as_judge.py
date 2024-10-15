import pandas as pd
from unitxt.operator import SequentialOperator
from unitxt.stream import MultiStream

# some toy input examples
test_examples = [
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports no foundation models",
        "contexts": [
            "Supported foundation models available with watsonx.ai. Watsonx.ai offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
        "a123123": "",
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Meta.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models, the most prominent are Llama, Mixtral and Granite.",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "never gonna give you up, never gonna let you down.",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
    },
]

# select the desired metrics.
# all available metrics are under "catalog.metrics.llm_as_judge.binary"
metric_names = [
    "answer_correctness_q_a_gt_loose_logprobs",
    "answer_correctness_q_a_gt_strict_logprobs",
    "faithfulness_q_c_a_logprobs",
    "faithfulness_c_a_logprobs",
    "context_relevance_q_c_ares_logprobs",
    "answer_relevance_q_a_logprobs",
]
metrics_path = "metrics.llm_as_judge.binary"

# select the desired model.
# all available models are under "catalog.engines.classification"
model_names = [
    "llama_3_1_70b_instruct_wml",
    "mixtral_8x7b_instruct_v01_wml",
    "gpt_4_turbo_openai",
]
models_path = "engines.classification"

if __name__ == "__main__":
    multi_stream = MultiStream.from_iterables({"test": test_examples}, copying=True)

    # to keep all results
    results = test_examples.copy()
    global_scores = {"question": "global"}

    for metric_name in metric_names:
        for model_name in model_names:
            # override the metric with the inference model
            llmaj_metric_name = f"{metrics_path}.{metric_name}[inference_model={models_path}.{model_name}]"

            # apply the metric over the input
            metrics_operator = SequentialOperator(steps=[llmaj_metric_name])
            instances = metrics_operator(multi_stream)["test"]
            instances = list(instances)

            # all scores will have this prefix
            score_name = f"{model_name}_{metric_name}"
            for i in range(len(instances)):
                results[i][score_name] = instances[i]["score"]["instance"][score_name]
                results[i][f"{score_name}_source"] = instances[i]["score"]["instance"][
                    f"{score_name}_judge_raw_input"
                ]
            global_scores[score_name] = instances[0]["score"]["global"][score_name]

    pd.DataFrame(results).transpose().to_csv("dataset_out.csv")
