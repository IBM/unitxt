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
        "metadata": {"data_classification_policy": ["public"]},
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
        "metadata": {"data_classification_policy": ["public"]},
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Meta.ai supports a variety of foundation models",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
        "metadata": {"data_classification_policy": ["public"]},
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "Watsonx.ai supports a variety of foundation models, the most prominent are Llama, Mixtral and Granite.",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
        "metadata": {"data_classification_policy": ["public"]},
    },
    {
        "question": "What foundation models are available in watsonx.ai ?",
        "answer": "never gonna give you up, never gonna let you down.",
        "contexts": [
            "Supported foundation models available with Meta. Meta AI offers numerous foundation models."
        ],
        "ground_truths": ["Many Large Language Models are supported by Watsonx.ai"],
        "metadata": {"data_classification_policy": ["public"]},
    },
]

# Select the desired metric(s).
# Each metric measures a certain aspect of the generated answer (answer_correctness, faithfulness,
# answer_relevance and context_relevance).
# All available metrics are under "catalog.metrics.rag.autorag.", ending with "judge"
# By default, all judges use llama_3_3_70b_instruct. We will soon see how to change this.
metric_names = [
    "metrics.rag.autorag.answer_correctness.llama_3_3_70b_instruct_wml_judge",
    "metrics.rag.autorag.faithfulness.llama_3_3_70b_instruct_wml_judge",
]

# select the desired model.
# all available models are under "catalog.engines.classification"
model_names = ["engines.classification.mixtral_8x7b_instruct_v01_wml"]

if __name__ == "__main__":
    multi_stream = MultiStream.from_iterables({"test": test_examples}, copying=True)

    # to keep all results
    results = test_examples.copy()
    global_scores = {"question": "global"}

    for metric_name in metric_names:
        for model_name in model_names:
            # override the metric with the inference model (to use a model different from the one in the metric name)
            llmaj_metric_name = f"{metric_name}[inference_model={model_name}]"

            # apply the metric over the input
            metrics_operator = SequentialOperator(steps=[llmaj_metric_name])
            instances = metrics_operator(multi_stream)["test"]
            instances = list(instances)

            score_name = instances[0]["score"]["instance"]["score_name"]
            for i in range(len(instances)):
                results[i][score_name] = instances[i]["score"]["instance"][score_name]
                results[i][f"{score_name}_source"] = instances[i]["score"]["instance"][
                    f"{score_name}_judge_raw_input"
                ]
            global_scores[score_name] = instances[0]["score"]["global"][score_name]

    pd.DataFrame(results).transpose().to_csv("dataset_out.csv")
