from unitxt import add_to_catalog
from unitxt.metrics import LlamaIndexCorrectness, LlamaIndexFaithfulness
from unitxt.test_utils.metrics import test_metric

# Test with mock
model_name = "mock"
model_name_normalized = model_name.replace(".", "_").replace("-", "_")

predictions = ["The right answer"]
references = [["The right answer", "The wrong answer"]]
task_data = [
    {
        "question": "question number 1",
        "contexts": ["context number 1"],
        # "reference_answers": ["The right answer", "The wrong answer"],
    },
]

metric_classes = {
    "correctness": LlamaIndexCorrectness,
    "faithfulness": LlamaIndexFaithfulness,
}

for metric_name, metric_class in metric_classes.items():
    metric = metric_class(model_name=model_name)

    score_name = f"{metric_name}_llama_index_by_{model_name_normalized}_judge"

    instance_targets = [
        {
            "score": 1.0,
            "score_name": score_name,
            score_name: 1.0,
            # "feedback": "The generated answer is fully correct and relevant to the user query, matching the reference answer exactly.",
        }
    ] * len(predictions)

    global_target = {
        "score": 1.0,
        "score_name": score_name,
        score_name: 1.0,
        "num_of_instances": 1,
    }

    outputs = test_metric(
        metric=metric,
        predictions=predictions,
        references=references,
        task_data=task_data,
        instance_targets=instance_targets,
        global_target=global_target,
    )

    # GPT model to catalog
    model_names = ["gpt-3.5-turbo", "mock"]
    for model_name in model_names:
        model_name_normalized = model_name.replace(".", "_").replace("-", "_")

        metric = (
            metric_class(model_name=model_name, data_classification_policy=["public"])
            if model_name != "mock"
            else metric_class(model_name=model_name)
        )

        add_to_catalog(
            metric,
            f"metrics.rag.{metric_name}.llama_index_by_{model_name_normalized}",
            overwrite=True,
        )
