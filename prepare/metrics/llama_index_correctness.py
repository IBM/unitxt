from src.unitxt import add_to_catalog
from src.unitxt.metrics import (
    LlamaIndexCorrectnessMetric,
)
from src.unitxt.test_utils.metrics import test_metric

metric = LlamaIndexCorrectnessMetric()

predictions = ["The right answer"]
references = [["The right answer"]]
inputs = [
    {
        "question": "question number 1",
        "contexts": ["context number 1"],
        "reference_answers": ["The right answer"],
    }
]

instance_targets = [  # nDCG is undefined at instance level
    {
        "score": 1.0,
        "score_name": "score",
        # "feedback": "The generated answer is fully correct and relevant to the user query, matching the reference answer exactly.",
    }
] * len(predictions)

global_target = {"score": 1.0, "score_name": "score"}

outputs = test_metric(
    metric=metric,
    predictions=predictions,
    references=references,
    additional_inputs=inputs,
    instance_targets=instance_targets,
    global_target=global_target,
)

add_to_catalog(metric, "metrics.rag.llama_index_correctness", overwrite=True)
